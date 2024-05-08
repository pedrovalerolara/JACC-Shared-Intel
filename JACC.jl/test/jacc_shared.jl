using JACC
using oneAPI

function convolution(i, j, input, ouput, filter, filter_size, num_inputs)
  for n in 1:num_inputs
    find = Int32(1)
    conv = Float32(0.0)  
    i_ind = Int32(i - filter_size)
    j_ind = Int32(j - filter_size)
    for fi in 1:filter_size
      for fj in 1:filter_size
        if (i_ind + fi > 0) && (j_ind + fj > 0)
          @inbounds conv += filter[find] * input[n, i_ind + fi, j_ind + fj] 
        end
        find += 1
      end
      find += 1
    end
    @inbounds ouput[n, i, j] = conv
  end
end

function convolution_shared(i, j, input, ouput, filter, filter_size, num_inputs)
  filter_shared = JACC.shared(filter)
  for n in 1:num_inputs
    find = Int32(1)
    conv = Float32(0.0)  
    i_ind = Int32(i - filter_size)
    j_ind = Int32(j - filter_size)
    for fi in 1:filter_size
      for fj in 1:filter_size
        if (i_ind + fi > 0) && (j_ind + fj > 0) 
          @inbounds conv += filter_shared[find] * input[n, i_ind + fi, j_ind + fj]
        end
        find += 1
      end
      find += 1
    end
    @inbounds ouput[n, i, j,] = conv
  end
end

SIZE_x = Int32(1024)
SIZE_y = Int32(1024)
NUM_inputs = Int32(256)
SIZE_filter = Int32(5)
input = ones(Float32, NUM_inputs, SIZE_x, SIZE_y)
output = ones(Float32, NUM_inputs, SIZE_x, SIZE_y)
filter = ones(Float32, SIZE_filter*SIZE_filter)
filter *= Float32(2.0)
jinput = JACC.Array(input)
joutput = JACC.Array(output)
joutput_shared = JACC.Array(output)
jfilter = JACC.Array(filter)
for n = 128:128:256
  for s = 512:512:1024  
    for f = 3:2:5
      @show n
      @show s
      @show f 
      @time begin
        JACC.parallel_for((s,s), convolution, jinput, joutput, jfilter, f, n)
      end
      @time begin
        JACC.parallel_for((s,s), convolution, jinput, joutput, jfilter, f, n)
      end
      @time begin
        JACC.parallel_for((s,s), convolution, jinput, joutput, jfilter, f, n)
      end
      @time begin
        JACC.parallel_for((s,s), convolution_shared, jinput, joutput_shared, jfilter, f, n)
      end
      @time begin
        JACC.parallel_for((s,s), convolution_shared, jinput, joutput_shared, jfilter, f, n)
      end

      @time begin
        JACC.parallel_for((s,s), convolution_shared, jinput, joutput_shared, jfilter, f, n)
      end

    end
  end
end


