import oneAPI
import JACC
using Test


@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "oneapi"
end

@testset "VectorAddLambda" begin

    function f(i, a)
        @inbounds a[i] += 5.0
    end

    N = 10
    dims = (N)
    a = round.(rand(Float32, dims) * 100)

    a_device = JACC.Array(a)
    JACC.parallel_for(N, f, a_device)

    a_expected = a .+ 5.0
    @test Array(a_device) ≈ a_expected rtol = 1e-5

end

@testset "AXPY" begin

    function axpy(i, alpha, x, y)
        @inbounds x[i] += alpha * y[i]
    end

    function seq_axpy(N, alpha, x, y)
        @inbounds for i in 1:N
            x[i] += alpha * y[i]
        end
    end

    N = 10
    # Generate random vectors x and y of length N for the interval [0, 100]
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha::Float32 = 2.5

    x_device = JACC.Array(x)
    y_device = JACC.Array(y)
    JACC.parallel_for(N, axpy, alpha, x_device, y_device)

    x_expected = x
    seq_axpy(N, alpha, x_expected, y)

    @test Array(x_device) ≈ x_expected rtol = 1e-1
end


@testset "JACC Shared" begin
  
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
  @time begin
    JACC.parallel_for((SIZE_x,SIZE_y), convolution, jinput, joutput, jfilter, SIZE_filter, NUM_inputs)
  end
  @time begin
    JACC.parallel_for((SIZE_x,SIZE_y), convolution, jinput, joutput_shared, jfilter, SIZE_filter, NUM_inputs)
  end
  #for n = 128:128:256
  #  for s = 512:512:1024  
  #    for f = 3:2:5
  #      @show n
  #      @show s
  #      @show f 
  #      @time begin
  #        JACC.parallel_for((s,s), convolution, jinput, joutput, jfilter, f, n)
  #      end
  #      @time begin
  #        JACC.parallel_for((s,s), convolution, jinput, joutput, jfilter, f, n)
  #      end
  #      @time begin
  #        JACC.parallel_for((s,s), convolution, jinput, joutput, jfilter, f, n)
  #      end
  #      @time begin
  #        JACC.parallel_for((s,s), convolution_shared, jinput, joutput_shared, jfilter, f, n)
  #      end
  #      @time begin
  #        JACC.parallel_for((s,s), convolution_shared, jinput, joutput_shared, jfilter, f, n)
  #      end
  #       @time begin
  #         JACC.parallel_for((s,s), convolution_shared, jinput, joutput_shared, jfilter, f, n)
  #       end
  #    end
  #  end
  #end
  @test joutput ≈ joutput_shared rtol = 1e-5
end


