module JACCCUDA

using JACC, CUDA

function JACC.parallel_for(N::I, f::F, x::Vararg{Union{<:Number,<:CuArray}}) where {I<:Integer,F<:Function}
  maxPossibleThreads = 512
  threads = min(N, maxPossibleThreads)
  blocks = ceil(Int, N / threads)
  CUDA.@sync @cuda threads = threads blocks = blocks shmem = numThreads * sizeof(Float64) _parallel_for_cuda(f, x...)
end

function JACC.parallel_for((M, N)::Tuple{I,I}, f::F, x::Vararg{Union{<:Number,<:CuArray}}) where {I<:Integer,F<:Function}
  numThreads = 32
  Mthreads = min(M, numThreads)
  Nthreads = min(N, numThreads)
  Mblocks = ceil(Int, M / Mthreads)
  Nblocks = ceil(Int, N / Nthreads)
  CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = (Mblocks, Nblocks) shmem = 4 * numThreads * numThreads * sizeof(Float64) _parallel_for_cuda_MN(f, x...)
end

function JACC.parallel_reduce(N::I, f::F, x::Vararg{Union{<:Number,<:CuArray}}) where {I<:Integer,F<:Function}
  numThreads = 512
  threads = min(N, numThreads)
  blocks = ceil(Int, N / threads)
  ret = CUDA.zeros(Float64, blocks)
  rret = CUDA.zeros(Float64, 1)
  CUDA.@sync @cuda threads = threads blocks = blocks shmem = numThreads * sizeof(Float64) _parallel_reduce_cuda(N, ret, f, x...)
  CUDA.@sync @cuda threads = threads blocks = 1 shmem = numThreads * sizeof(Float64) reduce_kernel_cuda(blocks, ret, rret)
  return rret
end


function JACC.parallel_reduce((M, N)::Tuple{I,I}, f::F, x::Vararg{Union{<:Number,<:CuArray}}) where {I<:Integer,F<:Function}
  numThreads = 16
  Mthreads = min(M, numThreads)
  Nthreads = min(N, numThreads)
  Mblocks = ceil(Int, M / Mthreads)
  Nblocks = ceil(Int, N / Nthreads)
  ret = CUDA.zeros(Float64, (Mblocks, Nblocks))
  rret = CUDA.zeros(Float64, 1)
  CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = (Mblocks, Nblocks) shmem = numThreads * numThreads * sizeof(Float64) _parallel_reduce_cuda_MN((M, N), ret, f, x...)
  CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = (1, 1) shmem = numThreads * numThreads * sizeof(Float64) reduce_kernel_cuda_MN((Mblocks, Nblocks), ret, rret)
  return rret
end

function _parallel_for_cuda(f, x...)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  f(i, x...)
  return nothing
end

function _parallel_for_cuda_MN(f, x...)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  f(i, j, x...)
  return nothing
end

function _parallel_reduce_cuda(N, ret, f, x...)
  shared_mem = @cuDynamicSharedMem(Float64, 512)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ti = threadIdx().x
  tmp::Float64 = 0.0
  shared_mem[ti] = 0.0

  if i <= N
    tmp = @inbounds f(i, x...)
    shared_mem[threadIdx().x] = tmp
  end
  sync_threads()
  if (ti <= 256)
    shared_mem[ti] += shared_mem[ti+256]
  end
  sync_threads()
  if (ti <= 128)
    shared_mem[ti] += shared_mem[ti+128]
  end
  sync_threads()
  if (ti <= 64)
    shared_mem[ti] += shared_mem[ti+64]
  end
  sync_threads()
  if (ti <= 32)
    shared_mem[ti] += shared_mem[ti+32]
  end
  sync_threads()
  if (ti <= 16)
    shared_mem[ti] += shared_mem[ti+16]
  end
  sync_threads()
  if (ti <= 8)
    shared_mem[ti] += shared_mem[ti+8]
  end
  sync_threads()
  if (ti <= 4)
    shared_mem[ti] += shared_mem[ti+4]
  end
  sync_threads()
  if (ti <= 2)
    shared_mem[ti] += shared_mem[ti+2]
  end
  sync_threads()
  if (ti == 1)
    shared_mem[ti] += shared_mem[ti+1]
    ret[blockIdx().x] = shared_mem[ti]
  end
  return nothing
end

function reduce_kernel_cuda(N, red, ret)
  shared_mem = @cuDynamicSharedMem(Float64, 512)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ii = i
  tmp::Float64 = 0.0
  if N > 512
    while ii <= N
      tmp += @inbounds red[ii]
      ii += 512
    end
  else
    tmp = @inbounds red[i]
  end
  shared_mem[i] = tmp
  sync_threads()
  if (i <= 256)
    shared_mem[i] += shared_mem[i+256]
  end
  sync_threads()
  if (i <= 128)
    shared_mem[i] += shared_mem[i+128]
  end
  sync_threads()
  if (i <= 64)
    shared_mem[i] += shared_mem[i+64]
  end
  sync_threads()
  if (i <= 32)
    shared_mem[i] += shared_mem[i+32]
  end
  sync_threads()
  if (i <= 16)
    shared_mem[i] += shared_mem[i+16]
  end
  sync_threads()
  if (i <= 8)
    shared_mem[i] += shared_mem[i+8]
  end
  sync_threads()
  if (i <= 4)
    shared_mem[i] += shared_mem[i+4]
  end
  sync_threads()
  if (i <= 2)
    shared_mem[i] += shared_mem[i+2]
  end
  sync_threads()
  if (i == 1)
    shared_mem[i] += shared_mem[i+1]
    ret[1] = shared_mem[1]
  end
  return nothing
end

function _parallel_reduce_cuda_MN((M, N), ret, f, x...)
  shared_mem = @cuDynamicSharedMem(Float64, 16 * 16)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  ti = threadIdx().x
  tj = threadIdx().y
  bi = blockIdx().x
  bj = blockIdx().y

  tmp::Float64 = 0.0
  shared_mem[((ti-1)*16)+tj] = tmp

  if (i <= M && j <= N)
    tmp = @inbounds f(i, j, x...)
    shared_mem[(ti-1)*16+tj] = tmp
  end
  sync_threads()
  if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+7)*16)+(tj+8)]
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+8)]
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+7)*16)+tj]
  end
  sync_threads()
  if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+3)*16)+(tj+4)]
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+4)]
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+3)*16)+tj]
  end
  sync_threads()
  if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+1)*16)+(tj+2)]
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+2)]
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+1)*16)+tj]
  end
  sync_threads()
  if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
    shared_mem[((ti-1)*16)+tj] += shared_mem[ti*16+(tj+1)]
    shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+1)]
    shared_mem[((ti-1)*16)+tj] += shared_mem[ti*16+tj]
    ret[bi, bj] = shared_mem[((ti-1)*16)+tj]
  end
  return nothing
end

function reduce_kernel_cuda_MN((M, N), red, ret)
  shared_mem = @cuDynamicSharedMem(Float64, 16 * 16)
  i = threadIdx().x
  j = threadIdx().y
  ii = i
  jj = j

  tmp::Float64 = 0.0
  shared_mem[(i-1)*16+j] = tmp

  if M > 16 && N > 16
    while ii <= M
      jj = threadIdx().y
      while jj <= N
        tmp = tmp + @inbounds red[ii, jj]
        jj += 16
      end
      ii += 16
    end
  elseif M > 16
    while ii <= N
      tmp = tmp + @inbounds red[ii, jj]
      ii += 16
    end
  elseif N > 16
    while jj <= N
      tmp = tmp + @inbounds red[ii, jj]
      jj += 16
    end
  elseif M <= 16 && N <= 16
    if i <= M && j <= N
      tmp = tmp + @inbounds red[i, j]
    end
  end
  shared_mem[(i-1)*16+j] = tmp
  red[i, j] = shared_mem[(i-1)*16+j]
  sync_threads()
  if (i <= 8 && j <= 8)
    if (i + 8 <= M && j + 8 <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i+7)*16)+(j+8)]
    end
    if (i <= M && j + 8 <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+8)]
    end
    if (i + 8 <= M && j <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i+7)*16)+j]
    end
  end
  sync_threads()
  if (i <= 4 && j <= 4)
    if (i + 4 <= M && j + 4 <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i+3)*16)+(j+4)]
    end
    if (i <= M && j + 4 <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+4)]
    end
    if (i + 4 <= M && j <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i+3)*16)+j]
    end
  end
  sync_threads()
  if (i <= 2 && j <= 2)
    if (i + 2 <= M && j + 2 <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i+1)*16)+(j+2)]
    end
    if (i <= M && j + 2 <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+2)]
    end
    if (i + 2 <= M && j <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i+1)*16)+j]
    end
  end
  sync_threads()
  if (i == 1 && j == 1)
    if (i + 1 <= M && j + 1 <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[i*16+(j+1)]
    end
    if (i <= M && j + 1 <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+1)]
    end
    if (i + 1 <= M && j <= N)
      shared_mem[((i-1)*16)+j] += shared_mem[i*16+j]
    end
    ret[1] = shared_mem[((i-1)*16)+j]
  end
  return nothing
end

function JACC.tile_size_x()
  return blockDim().x
end 

function JACC.tile_size_y()
  return blockDim().y
end 

function JACC.local_id_x(ind::I) where {I<:Integer}
  return threadIdx().x
end 

function JACC.local_id_y(ind::I) where {I<:Integer}
  return threadIdx().y
end 

function JACC.create_1Dtile()
  shmem = @cuDynamicSharedMem(Float64, blockDim().x)
  return shmem
end

#function JACC.shared(i::I, shmem, x::CuDeviceArray{T,N}) where {I<:Integer,T,N}
#  #shmem = @cuStaticSharedMem(T, blockDim().x)
#  i_global = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#  i_local = threadIdx().x 
#  @inbounds shmem[i_local] = 0.0 
#  if i == i_global
#    @inbounds shmem[i_local] = x[i_global]
#  elseif i != i_global
#    @inbounds shmem[i_local] = x[i + (i_local - 1)]
#  end
#  sync_threads()
#  return shmem
#end

function JACC.create_2Dtile()
  shmem = @cuDynamicSharedMem(Float64, blockDim().x * blockDim().y)
  return shmem
end

function JACC.shared((i, j)::Tuple{I,I}, shmem, x::CuDeviceArray{T,N}) where {I<:Integer,T,N}
  i_local = threadIdx().x
  j_local = threadIdx().y
  i_global = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j_global = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  if i == i_global && j == j_global
    tmp = x[i, j] 
  elseif i == i_global && j != j_global
    tmp = x[i, j + (j_local - 1)] 
  elseif i != i_global && j == j_global
    tmp = x[i + (i_local - 1), j]
  else
    tmp = x[i + (i_local - 1), j + (j_local - 1)]
  end
  shmem[(i_local - 1) * blockDim().x + j_local] = tmp
  sync_threads()
  return nothing
end

function JACC.global_shared((i, j)::Tuple{I,I}, x::CuDeviceArray{T,M,N}) where {I<:Integer,T,M,N}
  shmem = @cuDynamicSharedMem(T, blockDim().x * blockDim().y)
  i_local = threadIdx().x
  j_local = threadIdx().y
  i_global = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j_global = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  if i == i_global && j == j_global
    tmp = x[i, j] 
  elseif i == i_global && j != j_global
    tmp = x[i, j + (j_local - 1)] 
  elseif i != i_global && j == j_global
    tmp = x[i + (i_local - 1), j]
  else
    tmp = x[i + (i_local - 1), j + (j_local - 1)]
  end
  shmem[(i_local - 1) * blockDim().x + j_local] = tmp
  sync_threads()
  return shmem
end

function JACC.global_shared(x::CuDeviceArray{T,N}) where {T,N}
  if ndims(x) == 1
   shmem = @cuDynamicSharedMem(T, blockDim().x)
   i_local = threadIdx().x
   i_global = (blockIdx().x - 1) * blockDim().x + threadIdx().x
   @inbounds shmem[i_local] = x[i_global]
  elseif ndims(x) == 2 
   shmem = @cuDynamicSharedMem(T, blockDim().x * blockDim().y)
   i_local = threadIdx().x
   j_local = threadIdx().y
   i_global = (blockIdx().x - 1) * blockDim().x + threadIdx().x
   j_global = (blockIdx().y - 1) * blockDim().y + threadIdx().y
   @inbounds shmem[(i_local - 1) * blockDim().x + j_local] = x[i_global, j_global]
  end
  sync_threads()
  return shmem
end

#function JACC.local_shared(x::CuDeviceArray{T,N}) where {T,N}
#  shmem = @cuDynamicSharedMem(T, length(x))
#  if blockDim().y == 1
#    ind = threadIdx().x
#  else
#    i_local = threadIdx().x
#    j_local = threadIdx().y
#    ind = (i_local - 1) * blockDim().x + j_local
#  end
#  if ind <= length(x)
#    @inbounds shmem[ind] = x[ind] 
#  end
#  sync_threads()
#  return shmem
#end

function JACC.shared(x::CuDeviceArray{T,N}) where {T,N}
  size = length(x)
  shmem = @cuDynamicSharedMem(T, size)
  num_threads = blockDim().x * blockDim().y
  if (size <= num_threads)
    if blockDim().y == 1
      ind = threadIdx().x
      #if (ind <= size)
        @inbounds shmem[ind] = x[ind]
      #end
    else
      i_local = threadIdx().x
      j_local = threadIdx().y
      ind = (i_local - 1) * blockDim().x + j_local
      if ndims(x) == 1
        #if (ind <= size)
          @inbounds shmem[ind] = x[ind]
        #end
      elseif ndims(x) == 2
        #if (ind <= size)
          @inbounds shmem[ind] = x[i_local,j_local]
        #end
      end
    end
  else
    if blockDim().y == 1
      ind = threadIdx().x
      for i in blockDim().x:blockDim().x:size
        @inbounds shmem[ind] = x[ind]
        ind += blockDim().x
      end
    else
      i_local = threadIdx().x
      j_local = threadIdx().y
      ind = (i_local - 1) * blockDim().x + j_local
      if ndims(x) == 1
        for i in num_threads:num_threads:size
          @inbounds shmem[ind] = x[ind]
          ind += num_threads
        end
      elseif ndims(x) == 2
        for i in num_threads:num_threads:size
          @inbounds shmem[ind] = x[i_local,j_local]
          ind += num_threads
        end
      end  
    end
  end
  sync_threads()
  return shmem
end


function __init__()
  const JACC.Array = CUDA.CuArray{T,N} where {T,N}
end

end # module JACCCUDA
