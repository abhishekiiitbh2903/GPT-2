import os
import torch
from torch.distributed import init_process_group

def setup_device():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    else:
        # Vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return device, device_type, master_process, ddp_rank, ddp_local_rank, ddp_world_size,ddp


#For a system with 2 nodes and 4 GPUs per node:

# Node	Global Rank (ddp_rank)	Local Rank (ddp_local_rank) 	GPU Assigned
# Node 1	            0	                    0               	cuda:0
# Node 1	            1	                    1               	cuda:1
# Node 1	            2	                    2               	cuda:2
# Node 1	            3	                    3               	cuda:3
# Node 2	            4	                    0               	cuda:0
# Node 2	            5	                    1               	cuda:1
# Node 2	            6	                    2               	cuda:2
# Node 2	            7	                    3               	cuda:3
