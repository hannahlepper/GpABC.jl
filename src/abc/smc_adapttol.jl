
function initialiseABCSMC_at(input::SimulatedABCSMCInput;
        write_progress = true,
        progress_every = 1000,
        threshold
        )
    # the first run is an ABC rejection simulation
    rejection_input = SimulatedABCRejectionInput(input)

    rejection_output = ABCrejection_at(rejection_input;
                                    write_progress = write_progress,
                                    progress_every = progress_every,
                                    threshold)

    tracker =  SimulatedABCSMCTracker(input.n_params,
                             [rejection_output.n_accepted],
                             [rejection_output.n_tries],
                             [rejection_output.threshold],
                             rejection_output.n_accepted > 0,
                             [rejection_output.population],
                             [rejection_output.distances],
                             [rejection_output.weights],
                             input.priors,
                             input.distance_simulation_input,
                             input.max_iter,
                             1)

    return tracker
end

function ABCrejection_at(input::SimulatedABCRejectionInput;
    write_progress::Bool = true,
    progress_every::Int = 1000,
    threshold::Float64)

    numthreads = Threads.nthreads()
    filename = string(input.file_path, "pop1.jld2")

	checkABCInput(input)
    if write_progress
        @info "GpABC rejection simulation. ϵ = $threshold."
    end

	# initialise
    n_tries = 0
    n_accepted = 0
    accepted_parameters = zeros(input.n_particles, input.n_params)
    accepted_distances = zeros(input.n_particles)
    weight_values = zeros(input.n_particles)
    distance = 0.0

    
    # simulate
    while n_accepted < input.n_particles && n_tries < input.max_iter

        param_set = Array{Matrix{Float64}}(undef,numthreads) #needs to be an array of matrixes
        weight_value_set = zeros(numthreads)
        distance_set = zeros(numthreads)

        Threads.@threads for i in 1:numthreads
            out = Array{Any}(undef, 3)
            try
               out .= getparamsdist(input)
            catch e
                if isa(e, DimensionMismatch)
                    # This prevents the whole code from failing if there is a problem
                    # solving the differential equation(s). The exception is thrown by the
                    # distance function
                    @warn "The summarised simulated data does not have the same size as the summarised reference data. If this is not happening at every iteration it may be due to the behaviour of OrdinaryDiffEq::solve - please check for related warnings. Continuing to the next iteration."
                    #n_tries += 1
                    continue
                else
                    throw(e)
                end
            end
           
            param_set[i] = out[1]
            weight_value_set[i] = out[2]
            distance_set[i] = out[3]

        end

        n_tries += numthreads

        for i in 1:numthreads
            if n_accepted < input.n_particles
                if distance_set[i] <= threshold
                    n_accepted += 1
                    accepted_parameters[n_accepted,:] = param_set[i]
                    accepted_distances[n_accepted] = distance_set[i]
                    weight_values[n_accepted] = weight_value_set[i]
                end
            end
        end

        save_results = [n_tries, n_accepted, accepted_parameters, accepted_distances, weight_values, threshold]
        @save filename save_results

        if write_progress && (n_tries % progress_every == 0)
            @info "GpABC rejection simulation. Accepted $(n_accepted)/$(n_tries) particles."
        end
    end

    if n_accepted < input.n_particles
        @warn "Simulation reached maximum iterations $(input.max_iter) before finding $(input.n_particles) particles - will return $n_accepted"
        accepted_parameters = accepted_parameters[1:n_accepted, :]
        accepted_distances = accepted_distances[1:n_accepted]
        weight_values = weight_values[1:n_accepted]
    end


    # output
    output = SimulatedABCRejectionOutput(input.n_params,
                                n_accepted,
                                n_tries,
                                threshold,
                                accepted_parameters,
                                accepted_distances,
                                StatsBase.Weights(weight_values ./ sum(weight_values)) # normalise weights
                                )

    return output

end

function initialise_threshold(input::SimulatedABCSMCInput, kmax::Float64)

    numthreads = Threads.nthreads()
    max = round(Int, kmax*input.n_particles)
    parameters = zeros(max, input.n_params)
    distances = fill(Inf, max)
    k = 1

    _input = SimulatedABCRejectionInput(input)

    @info "initialising threshold. Number of runs: $max"

    while k <= max
        Threads.@threads for i in 1:numthreads
            kthread = k + i - 1
            if kthread <= max
                out = Array{Any}(undef, 3)
                try
                    out .= getparamsdist(_input)
                catch e
                    if isa(e, DimensionMismatch)
                        # This prevents the whole code from failing if there is a problem
                        # solving the differential equation(s). The exception is thrown by the
                        # distance function
                        @warn "The summarised simulated data does not have the same size as the summarised reference data. If this is not happening at every iteration it may be due to the behaviour of OrdinaryDiffEq::solve - please check for related warnings. Continuing to the next iteration."
                        #n_tries += 1
                        continue
                    else
                        throw(e)
                    end
                end
                
                parameters[kthread,:] = out[1] 
                distances[kthread] = out[3]
                
            end

        end
        k += numthreads
    end

    valid = isfinite.(distances)
    lowestdistances = partialsort(distances[valid], 1:input.n_particles)
    highestdist = maximum(lowestdistances)
    @info "Starting distance: $highestdist"
    return highestdist, parameters[valid, :]
end

function resample_particles(particles, weights)
    n = size(particles, 1)
    weights_norm = StatsBase.Weights(weights ./ sum(weights))
    resampled_indexes = sample(1:n, weights_norm, n, replace=true)
    return particles[resampled_indexes,:]
end

function estimatect(particles_t, particles_tmin1)
    x_nu = collect(eachrow(particles_t))
    x_de = collect(eachrow(particles_tmin1))

    r = densratiofunc(x_nu, x_de, KLIEP(), optlib=OptimLib)
    ct = maximum(r(x) for x in x_nu) #maximum over current generation

    # Guard against degenerate cases
    if !isfinite(ct) || ct <= 0.0
        @warn "Density ratio estimation returned invalid ct=$ct"
    end
    return ct
end

#for 1st population
function adapt_threshold(particles_t, priordraws, weights_t)
    resamp_par_t = resample_particles(particles_t, weights_t)

    ct = estimatect(resamp_par_t, priordraws)

    qt = 1/ct
    @info "ct: $ct, qt: $qt"

    return qt
end

function adapt_threshold(particles_t, particles_tmin1, weights_t, weights_tmin1)
    resamp_par_t = resample_particles(particles_t, weights_t)
    resamp_par_tmin1 = resample_particles(particles_tmin1, weights_tmin1)

    ct = estimatect(resamp_par_t, resamp_par_tmin1)

    qt = 1/ct
    @info "ct: $ct, qt: $qt"

    return qt
end


function ABCSMC_at(
        input::T;
        write_progress = true,
        progress_every = 1000,
        k=2.,
        min_populations=3,
        max_populations=20,
        max_qt=0.99
        ) where {T<:ABCSMCInput}

    #initialise tolerance threshold
    threshold, priordraws = initialise_threshold(input, k)

    #run rejection from priors (first population) and get tracker
    tracker = initialiseABCSMC_at(input; write_progress=write_progress,
                progress_every=progress_every, threshold=threshold)
    
    #initialise adaptive quantile
    qt = 0.5
    q_history = Float64[]

    if tracker.can_continue

        #get first adaptive threshold
        q = adapt_threshold(tracker.population[end], priordraws, tracker.weights[end])
        push!(q_history, qt)
        if !(isfinite(qt))
            @info "qt=$qt: stopping simulation. "
        else
            threshold = quantile(tracker.distances[end], qt)
            @info "Next population distance quantile: $qt"

            while (qt <= max_qt || tracker.pop_n <= min_populations) && tracker.pop_n < max_populations

                iterateABCSMC!(tracker, threshold, input.n_particles, input.file_path;
                    write_progress = write_progress,
                    progress_every = progress_every)

                if !tracker.can_continue
                    break
                end

                qt = adapt_threshold(tracker.population[end], tracker.population[end-1], tracker.weights[end], tracker.weights[end-1])
                push!(q_history, qt)

                if !(isfinite(qt))
                    @info "qt=$qt: stopping simulation. "
                    break
                end

                threshold = quantile(tracker.distances[end], qt)
                @info "Next population distance quantile: $qt"

                file_name = string(input.file_path, "qhist.jld2")
                @save file_name q_history

            end
        end
    else
        @warn "No particles selected at initial rejection ABC step of simulated SMC ABC - terminating algorithm"
    end

    return buildAbcSmcOutput(tracker), q_history
end

function SimulatedABCSMC_at(reference_data::AbstractArray{AF,2},
    simulator_function::Function,
    priors::AbstractArray{D,1},
    threshold_schedule::AbstractArray{AF,1},
    n_particles::Int,
    file_path::String;
    summary_statistic::Union{String,AbstractArray{String,1},Function} = "keep_all",
    distance_function::Union{Function,Metric}=Distances.euclidean,
    max_iter::Int=10 * n_particles,
    k::Float64=2.,
    max_populations::Int=20,
    min_populations::Int=3,
    max_qt::Float64=0.99,
    kwargs...
    ) where {
    AF<:AbstractFloat,
    D<:ContinuousUnivariateDistribution
    }

    n_params = length(priors)

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)
    distance_simulation_input = DistanceSimulationInput(reference_summary_statistic, simulator_function, summary_statistic, distance_function)
    input = SimulatedABCSMCInput(n_params, n_particles, threshold_schedule,
                                    priors, distance_simulation_input,
                                    max_iter, file_path)

    return ABCSMC_at(input; k=k, min_populations=min_populations, max_populations=max_populations, max_qt=max_qt, kwargs...)

end