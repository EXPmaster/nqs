import pickle
import numpy as np
import torch
import cuquantum
from cuquantum import cutensornet as cutn
import tensornetwork as tn


def sample_mps(mps_tensors: list[torch.Tensor], num_samples: int) -> torch.Tensor:
    device = mps_tensors[0].device
    stream = torch.cuda.current_stream()
    num_qubits = len(mps_tensors)
    handle = cutn.create()
    data_type = cuquantum.cudaDataType.CUDA_C_32F

    mps_tensor_extents = []
    mps_tensor_strides = []
    mps_tensor_ptrs = []
    target_tensor_ptrs = []
    for i in range(num_qubits):
        tensor = mps_tensors[i]
        mps_tensor_extents.append(tensor.size())
        mps_tensor_ptrs.append(tensor.data_ptr())
        mps_tensor_strides.append([stride for stride in tensor.stride()])
    
    quantum_state = cutn.create_state(
        handle,
        cutn.StatePurity.PURE,
        num_qubits,
        (2, ) * num_qubits,
        data_type
    )
    # Fortran order
    samples = np.empty((num_qubits, num_samples), dtype='int64', order='F')
    cutn.state_initialize_mps(
        handle,
        quantum_state,
        cutn.BoundaryCondition.OPEN,
        mps_tensor_extents,
        mps_tensor_strides,
        mps_tensor_ptrs
    )

    # cutn.state_finalize_mps(handle, quantum_state, cutn.BoundaryCondition.OPEN, mps_tensor_extents, mps_tensor_strides)
    work_desc = cutn.create_workspace_descriptor(handle)

    # use 2GB of the totol free size
    scratch_size = 10 * (10 ** 9)
    scratch_space = torch.cuda.caching_allocator_alloc(scratch_size, device, stream)

    # cutn.state_prepare(handle, quantum_state, scratch_size, work_desc, stream.cuda_stream)
    # extents_out, strides_out = cutn.state_compute(handle, quantum_state, work_desc, mps_tensor_ptrs, stream.cuda_stream)

    # Create the quantum circuit sampler
    sampler = cutn.create_sampler(handle, quantum_state, num_qubits, 0)
    

    # Configure the quantum circuit sampler with hyper samples for the contraction optimizer
    num_hyper_samples_dtype = cutn.sampler_get_attribute_dtype(cutn.SamplerAttribute.CONFIG_NUM_HYPER_SAMPLES)
    num_hyper_samples = np.asarray(8, dtype=num_hyper_samples_dtype)
    cutn.sampler_configure(handle, sampler, 
        cutn.SamplerAttribute.CONFIG_NUM_HYPER_SAMPLES, 
        num_hyper_samples.ctypes.data, num_hyper_samples.dtype.itemsize)

    # Prepare the quantum circuit sampler
    cutn.sampler_prepare(handle, sampler, scratch_size, work_desc, stream.cuda_stream)
    # print("Prepared the specified quantum circuit state sampler")

    flops_dtype = cutn.sampler_get_attribute_dtype(cutn.SamplerAttribute.INFO_FLOPS)
    flops = np.zeros(1, dtype=flops_dtype)
    cutn.sampler_get_info(handle, sampler, cutn.SamplerAttribute.INFO_FLOPS, flops.ctypes.data, flops.dtype.itemsize)
    print(f"Total flop count for sampling = {flops.item()/1e9} GFlop")

    workspace_size_d = cutn.workspace_get_memory_size(handle, 
        work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)

    if workspace_size_d <= scratch_size:
        cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, scratch_space, workspace_size_d)
    else:
        print("Error:Insufficient workspace size on Device")
        cutn.destroy_workspace_descriptor(work_desc)
        cutn.destroy_sampler(sampler)
        cutn.destroy_state(quantum_state)
        cutn.destroy(handle)
        torch.cuda.caching_allocator_delete(scratch_space)
        print("Free resource and exit.")
        exit()

    # print("Set the workspace buffer for sampling")
    # Sample the quantum circuit state
    cutn.sampler_sample(handle, sampler, num_samples, work_desc, samples.ctypes.data, stream.cuda_stream)
    stream.synchronize()
    # print("Performed quantum circuit state sampling")
    # print("Bit-string samples:")
    # hist = np.unique(samples.T, axis=0, return_counts=True)
    # for bitstring, count in zip(*hist):
    #     bitstring = np.array2string(bitstring, separator='')[1:-1]
    #     print(f"{bitstring}: {count}")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_sampler(sampler)
    cutn.destroy_state(quantum_state)
    cutn.destroy(handle)
    torch.cuda.caching_allocator_delete(scratch_space)
    # print("Free resource and exit.")
    # return torch.as_tensor(samples.T, dtype=torch.int32, device=device)
    with open('samples_ising_50q_-83.4122.pkl', 'wb') as f:
        pickle.dump(samples.T, f)
    return samples.T


if __name__ == '__main__':
    # num_qubits = 50
    # state = []
    # for i in range(num_qubits):
    #     if i == 0:
    #         shape = (2, 16)
    #     elif i == num_qubits - 1:
    #         shape = (16, 2)
    #     else:
    #         shape = (16, 2, 16)
    #     state.append(torch.randn(*shape, dtype=torch.complex64, device='cuda'))
    with torch.device('cuda'):
        state = tn.FiniteMPS.load(
            'ground_states/Ising/50qubits/state_-83.4122_-1.500.pkl',
            backend='pytorch'
        )
        # zero_state = [
        #     np.array(
        #         [1/np.sqrt(2), 1/np.sqrt(2)],
        #         dtype=np.complex64
        #     ).reshape(1, -1, 1)
        #     for k in range(len(state.tensors)) 
        # ]
        # zero_state = [
        #     np.array(
        #         [1, 0],
        #         dtype=np.complex64
        #     ).reshape(1, -1, 1)
        #     if k % 2 == 0 else
        #     np.array(
        #         [0, 1],
        #         dtype=np.complex64
        #     ).reshape(1, -1, 1)
        #     for k in range(len(state.tensors)) 
        # ]
        
        # zero_state = tn.FiniteMPS(zero_state, backend='pytorch')
        # # print(state.fidelity(zero_state))
        # state = zero_state
        
        state.tensors[0] = state.tensors[0].squeeze(0)
        state.tensors[-1] = state.tensors[-1].squeeze(-1)
        state = [s.contiguous() for s in state.tensors]

    output_samples = sample_mps(state, 2048)
    # print(bitstrings[0:10])
    # print(output_samples)
