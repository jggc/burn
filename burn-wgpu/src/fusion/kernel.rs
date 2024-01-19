use crate::compute::{compute_client, Kernel};
use crate::fusion::strides_dyn_rank;
use crate::fusion::WgpuFusionHandle;
use crate::{FloatElement, GraphicsApi, IntElement, Wgpu};
use burn_fusion::stream::Context;
use burn_fusion::TensorDescription;
use burn_tensor::Device;

/// Many kernels can be used for the same set of tensor operations fused into one.
///
/// This type makes it easy to group those potential kernels and execute the best one depending on
/// the context.
#[derive(new)]
pub struct FusionKernelSet {
    kernels: Vec<Box<dyn FusionKernel>>,
}

/// The priority of a kernel.
pub enum Priority {
    /// When a kernel can be executed in the specified context with its priority, higher is better.
    Available(u8),
    /// When a kernel can't be executed in the specified context.
    Unavailable,
}

#[derive(new)]
pub struct SelectedKernel {
    kernel: Box<dyn Kernel>,
    info: Vec<OutputInfo>,
}

// Information related to the output of this kernel.
pub enum OutputInfo {
    Inplace { input_index: usize },
    Array { size: usize },
}

pub trait FusionKernel: Send + Sync {
    /// Returns the priority of this kernel based on the input and output information.
    fn priority(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> Priority;
    /// Returns a [selected kernel](SelectedKernel) that can be executed by the compute server.
    fn kernel(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> SelectedKernel;
}

impl FusionKernelSet {
    /// Execute the best kernel based on the given information.
    pub fn execute<G: GraphicsApi, F: FloatElement, I: IntElement>(
        &self,
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
        scalars_f32: usize,
        scalars_i32: usize,
        context: &mut Context<'_, Wgpu<G, F, I>>,
        device: Device<Wgpu<G, F, I>>,
    ) {
        let client = compute_client::<G>(&device);

        let (handles_input, inputs_description_updated, outputs_description_updated) =
            process_inputs_outputs(inputs, outputs, context);

        let selected = self.select_kernel(
            &handles_input,
            &inputs_description_updated,
            &outputs_description_updated,
        );

        let mut info =
            Vec::with_capacity((inputs.len() + outputs.len()) * inputs[0].shape.len() * 2);
        let mut handles = Vec::with_capacity(inputs.len() + outputs.len() + 2);
        let mut output_register = Vec::with_capacity(outputs_description_updated.len());

        // We register the info and handles for the inputs.
        for (handle, tensor) in handles_input.into_iter().zip(inputs_description_updated) {
            register_info_tensor(&mut info, tensor, &handle);
            handles.push(handle.handle);
        }

        // We register the info and handles for the outputs.
        for (tensor, output_info) in outputs_description_updated
            .into_iter()
            .zip(selected.info.iter())
        {
            match output_info {
                // Use the input inplace for this output.
                OutputInfo::Inplace { input_index } => {
                    let handle = handles.get(*input_index).unwrap().clone();
                    let handle_fusion = WgpuFusionHandle {
                        client: client.clone(),
                        device: device.clone(),
                        strides: strides_dyn_rank(&tensor.shape),
                        handle,
                    };
                    output_register.push((tensor.id, handle_fusion));
                }
                // Create a new buffer for this output.
                OutputInfo::Array { size } => {
                    let handle_fusion = WgpuFusionHandle {
                        client: client.clone(),
                        device: device.clone(),
                        strides: strides_dyn_rank(&tensor.shape),
                        handle: client.empty(*size),
                    };

                    register_info_tensor(&mut info, tensor, &handle_fusion);
                    handles.push(handle_fusion.handle.clone());
                    output_register.push((tensor.id, handle_fusion));
                }
            };
        }

        // Create the info buffer.
        handles.push(client.create(bytemuck::cast_slice(&info)));

        // Finally we finish with the named bindings.
        if scalars_f32 > 0 {
            handles
                .push(client.create(bytemuck::cast_slice(&context.scalar_floats[0..scalars_f32])));
        }

        if scalars_i32 > 0 {
            handles.push(client.create(bytemuck::cast_slice(&context.scalar_ints[0..scalars_i32])));
        }

        // We have to register the output handles to the context.
        for (id, handle) in output_register {
            context.handles.register_handle(id, handle);
        }

        // Execute the kernel.
        client.execute(selected.kernel, &handles.iter().collect::<Vec<_>>());
    }

    fn select_kernel(
        &self,
        handles_input: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> SelectedKernel {
        // For now we simply select the kernel with the highest priority.
        let mut selected = self
            .kernels
            .iter()
            .filter_map(
                |source| match source.priority(handles_input, inputs, outputs) {
                    Priority::Available(priority) => Some((source, priority)),
                    Priority::Unavailable => None,
                },
            )
            .collect::<Vec<_>>();

        selected.sort_by(|(_, priority_a), (_, priority_b)| priority_a.cmp(priority_b));

        let selected = selected.pop().unwrap().0;

        selected.kernel(handles_input, inputs, outputs)
    }
}

fn register_info_tensor(
    info: &mut Vec<u32>,
    tensor: &TensorDescription,
    handle: &WgpuFusionHandle,
) {
    if info.is_empty() {
        info.push(handle.strides.len() as u32);
    }

    for s in handle.strides.iter() {
        info.push(*s as u32);
    }
    for s in tensor.shape.iter() {
        info.push(*s as u32);
    }
}

pub fn process_inputs_outputs<'a, G: GraphicsApi, F: FloatElement, I: IntElement>(
    inputs: &[&TensorDescription],
    outputs: &[&TensorDescription],
    context: &'a mut Context<'_, Wgpu<G, F, I>>,
) -> (
    Vec<WgpuFusionHandle>,
    Vec<&'a TensorDescription>,
    Vec<&'a TensorDescription>,
) {
    let mut inputs_description_updated = Vec::with_capacity(inputs.len());
    let mut outputs_description_updated = Vec::with_capacity(outputs.len());
    let mut handles_input = Vec::new();

    for tensor in inputs.iter() {
        let status = &tensor.status; // Important to take the status of the relative graph and not
                                     // the global graph, since the status of the global graph
                                     // might be of a later operation on the same tensor id.
        let tensor = context.tensors.get(&tensor.id).unwrap();
        let handle = context.handles.get_handle(&tensor.id, status);

        handles_input.push(handle);
        inputs_description_updated.push(tensor);
    }

    for tensor in outputs.iter() {
        let tensor = context.tensors.get(&tensor.id).unwrap();
        outputs_description_updated.push(tensor);
    }

    (
        handles_input,
        inputs_description_updated,
        outputs_description_updated,
    )
}
