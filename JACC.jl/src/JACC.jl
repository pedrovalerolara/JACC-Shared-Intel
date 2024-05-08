module JACC
using StaticArrays, Setfield

# module to set back end preferences 
include("JACCPreferences.jl")

export Array
export parallel_for

global Array

function parallel_for(N::I, f::F, x::Vararg{Union{<:Number,<:Base.Array}}) where {I<:Integer,F<:Function}
  Threads.@threads :static for i in 1:N
    f(i, x...)
  end
end

function parallel_for((M, N)::Tuple{I,I}, f::F, x::Vararg{Union{<:Number,<:Base.Array}}) where {I<:Integer,F<:Function}
  Threads.@threads :static for j in 1:N
    for i in 1:M
      f(i, j, x...)
    end
  end
end

function parallel_reduce(N::I, f::F, x::Vararg{Union{<:Number,<:Base.Array}}) where {I<:Integer,F<:Function}
  tmp = zeros(Threads.nthreads())
  ret = zeros(1)
  Threads.@threads :static for i in 1:N
    tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, x...)
  end
  for i in 1:Threads.nthreads()
    ret = ret .+ tmp[i]
  end
  return ret
end

function parallel_reduce((M, N)::Tuple{I,I}, f::F, x::Vararg{Union{<:Number,<:Base.Array}}) where {I<:Integer,F<:Function}
  tmp = zeros(Threads.nthreads())
  ret = zeros(1)
  Threads.@threads :static for j in 1:N
    for i in 1:M
      tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, j, x...)
    end
  end
  for i in 1:Threads.nthreads()
    ret = ret .+ tmp[i]
  end
  return ret
end

function tile_size_x()
  return 256
end

function tile_size_y()
  return 256
end

function local_id_x(ind::I) where {I<:Integer} 
  return ind
end

function local_id_y(ind::I) where {I<:Integer} 
  return Threads.threadid()
end

function local_id_y(ind::I, x::Base.Array{T,N}) where {I<:Integer, T,N} 
  return Threads.threadid()
end

function create_1Dtile()
  shmem = Core.Array{Float64, 256}
  return 
end

function create_2Dtile()
  shmem = Core.Array{Float64, 256*256}
  return 
end

function shared()
  @info("Playing with JACC.shared")
end

function global_shared(x::Base.Array{T,N}) where {T,N}
#  if ndims(x) == 1
#    size = length(x)
#    i_local = 0
#    shmem = zeros(T,ceil(Int, size / Threads.threadpoolsize()))
#    Threads.@threads :static for i in 1:size
#      @inbounds shmem[i_local] = x[i]
#      i_local += 1
#    end
#  elseif ndims(x) == 2 
#    size_m = size(x,1)
#    size_n = size(x,2)
#    j_local = 0
#    shmem = zeros(T,(size_m, ceil(Int, size_n / Threads.threadpoolsize()) 
#    Threads.@threads :static for j in 1:size_n
#      for i in 1:size_m
#        @inbounds shmem[i,j_local] = x[i,j]
#        j_local += 1
#      end
#    end  
#  end
#  return shmem
  @info("Playing with JACC.shared")
end

function shared(x::Base.Array{T,N}) where {T,N}
  #n = length(x)
  #shmem = zeros(T, n)
  #@show n
  #sshmem = SVector{size,T}
  #sshmem = @SVector zeros(T,size)
  #sshmem = SVector{size,T}(1:size)
  #if ndims(x) == 1
  #  for i in 1:n
  #    @inbounds shmem[i] = x[i]
  #  end
  #elseif ndims(x) == 2
  #  ind = 0
  #  for j in 1:size(x,2)
  #    for i in 1:size(x,1)
  #      @inbounds shmem[ind] = x[i,j]
  #    end
  #  end
  #end
  return x
end

function __init__()
  @info("Using JACC backend: $(JACCPreferences.backend)")

  if JACCPreferences.backend == "threads"
    const JACC.Array = Base.Array{T,N} where {T,N}
  end
end

end # module JACC
