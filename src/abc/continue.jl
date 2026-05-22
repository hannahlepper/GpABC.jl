#function for continuing a simulation when given necessary components
#Continue: threshold schedule
#First need to reconstruct the fit object, either from an existing object or from the components of one

#take an existing fit object
function SimulatedABCSMC_continue(
    oldoutput::SimulatedABCSMCOutput,
    reference_data::AbstractArray{AF,2},
    simulator_function::Function,
    priors::AbstractArray{D,1},
    threshold_schedule::AbstractArray{AF,1},
    n_particles::Int,
    file_path::String;
    summary_statistic::Union{String,AbstractArray{String,1},Function} = "keep_all",
    distance_function::Union{Function,Metric}=Distances.euclidean,
    max_iter::Int=10 * n_particles,
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

    n_pops_so_far = length(oldoutput.n_tries)

    tracker = SimulatedABCSMCTracker(input.n_params,
                            oldoutput.n_accepted,
                            oldoutput.n_tries,
                            input.threshold_schedule[1:n_pops_so_far],
                            oldoutput.n_accepted[end] > 0,
                            oldoutput.population,
                            oldoutput.distances,
                            oldoutput.weights,
                            input.priors,
                            input.distance_simulation_input,
                            input.max_iter,
                            n_pops_so_far)

    return ABCSMC_continue(tracker,input; kwargs...)

end

function SimulatedABCSMC_at_continue(
    oldoutput::SimulatedABCSMCOutput,
    reference_data::AbstractArray{AF,2},
    simulator_function::Function,
    priors::AbstractArray{D,1},
    threshold_schedule::AbstractArray{AF,1},
    n_particles::Int,
    file_path::String,
    qhist::Vector{Float64};
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

    n_pops_so_far = length(oldoutput.n_tries)

    tracker = SimulatedABCSMCTracker(input.n_params,
                            oldoutput.n_accepted,
                            oldoutput.n_tries,
                            input.threshold_schedule,
                            oldoutput.n_accepted[end] > 0,
                            oldoutput.population,
                            oldoutput.distances,
                            oldoutput.weights,
                            input.priors,
                            input.distance_simulation_input,
                            input.max_iter,
                            n_pops_so_far)

    return ABCSMC_at_continue(tracker, input, qhist; k=k, min_populations=min_populations, max_populations=max_populations, max_qt=max_qt, kwargs...)

end

function resumeABCSMC!(tracker::SimulatedABCSMCTracker,
        threshold::AbstractFloat,
        n_toaccept::Int,
        file_path::String;
        write_progress = true,
        progress_every = 1000)
    
    numthreads = Threads.nthreads()
    file_name = string(file_path, "pop", tracker.pop_n, ".jld2")
    
    if write_progress
        @info "GpABC SMC simulation ϵ = $threshold"
    end
    if threshold > tracker.threshold_schedule[end]
        @warn "current threshold less strict than previous one."
    end

    # (re-) initialise
    kernels = generate_kernels(tracker.population[end-1], tracker.priors)
    population = deepcopy(tracker.population[end])
    distances = deepcopy(tracker.distances[end])
    weight_values = copy(tracker.weights[end].values)
    n_tries = tracker.n_tries[end]
    n_accepted = tracker.n_accepted[end]

    # simulate
    while n_accepted < n_toaccept && n_tries < tracker.max_iter

        param_set = Array{Matrix{Float64}}(undef,numthreads) #needs to be an array of matrixes
        weight_value_set = zeros(numthreads)
        distance_set = zeros(numthreads)

        #parameters, weight_value = generate_parameters(tracker.priors, tracker.weights[end], kernels)
        # run simulation for a single particle
        
        Threads.@threads for i in 1:numthreads
            out = Array{Any}(undef, 3)
            try
                out = getparamsdist(tracker, kernels)
                #distance = simulate_distance(parameters, tracker.distance_simulation_input)
            catch e
                if isa(e, DimensionMismatch)
                    # This prevents the whole code from failing if there is a problem
                    # solving the differential equation(s). The exception is thrown by the
                    # distance function
                    @warn "The summarised simulated data does not have the same size as the summarised reference data. If this is not happening at every iteration it may be due to the behaviour of OrdinaryDiffEq::solve - please check for related warnings. Continuing to the next iteration."
                    n_tries += 1
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

        # Handle result
        for i in 1:numthreads
            if n_accepted < n_toaccept
                if distance_set[i] <= threshold
                    n_accepted += 1
                    population[n_accepted,:] = param_set[i]
                    distances[n_accepted] = distance_set[i]
                    weight_values[n_accepted] = weight_value_set[i]
                end
            end
        end

        save_results = [n_tries, n_accepted, population, distances, weight_values, threshold]
        @save file_name save_results

        if write_progress && (n_tries % progress_every == 0)
            @info "GpABC SMC simulation accepted $(n_accepted)/$(n_tries) particles."
        end
    end

    if n_accepted == 0
        @warn "Simulation reached maximum $(tracker.max_iter) iterations without selecting any particles"
    elseif n_accepted < n_toaccept
        population = population[1:n_accepted, :]
        weight_values = weight_values[1:n_accepted]
        distances = distances[1:n_accepted]
        @warn "Simulation reached maximum $(tracker.max_iter) iterations before finding $(n_toaccept) particles - will return $n_accepted"
    end

    pop!(tracker.n_accepted); pop!(tracker.n_tries); pop!(tracker.threshold_schedule)
    pop!(tracker.population); pop!(tracker.distances); pop!(tracker.weights)

    update_smctracker!(tracker, n_accepted, n_tries, threshold,
                        population, distances, weight_values)
end


function ABCSMC_continue(
        tracker::SimulatedABCSMCTracker,
        input::SimulatedABCSMCInput;
        write_progress = true,
        progress_every = 1000,
        ) 
    
    n_tries = tracker.n_tries
    n_accepted = tracker.n_accepted
    n_pops_so_far = length(n_tries)
    last_pop_complete = !(n_accepted[end] < input.n_particles && n_tries[end] < tracker.max_iter)

    if last_pop_complete #last population is complete
        go_to_next = true
    else #last population is not complete, resume from this point
        go_to_next = false
    end

    if tracker.can_continue
        if !go_to_next
            threshold = input.threshold_schedule[n_pops_so_far]
            resumeABCSMC!(tracker, threshold, input.n_particles, input.file_path;
                write_progress = write_progress,
                progress_every = progress_every)
            go_to_next = true
        end
    end
    
    if tracker.can_continue
        for i in (n_pops_so_far+1):length(input.threshold_schedule)
            threshold = input.threshold_schedule[i]
            iterateABCSMC!(tracker, threshold, input.n_particles, input.file_path;
                write_progress = write_progress,
                progress_every = progress_every)
            if !tracker.can_continue
                break
            end
        end
    else
        @warn "No particles selected at initial rejection ABC step of simulated SMC ABC - terminating algorithm"
    end

    return buildAbcSmcOutput(tracker)
end

function ABCSMC_at_continue(
        tracker::SimulatedABCSMCTracker,
        input::SimulatedABCSMCInput,
        qhist::Vector{Float64};
        write_progress = true,
        progress_every = 1000,
        k=2.,
        min_populations=3,
        max_populations=20,
        max_qt=0.99
        ) 
    
    n_tries = tracker.n_tries
    n_accepted = tracker.n_accepted
    n_pops_so_far = length(n_tries)
    last_pop_complete = !(n_accepted[end] < input.n_particles && n_tries[end] < tracker.max_iter)
    q_history = qhist

    if n_pops_so_far < 2
        @info "can't continue until have done initialisation step and first population. "
        return
    else
        if last_pop_complete #last population is complete
            qt = adapt_threshold(tracker.population[end], tracker.population[end-1], tracker.weights[end], tracker.weights[end-1])
                push!(q_history, qt)

            if !(isfinite(qt))
                @info "qt=$qt: stopping simulation. "
                return
            end

            threshold = quantile(tracker.distances[end], qt)
            @info "Next population distance quantile: $qt"
            
            go_to_next = true
    
        else #last population is not complete, resume from this point
            qt = last(q_history)
            threshold = quantile(tracker.distances[end-1], qt) #take quantile of last complete population
            go_to_next = false
        end
    end

    if tracker.can_continue

        while (qt <= max_qt || tracker.pop_n <= min_populations) && tracker.pop_n < max_populations
            
            if !go_to_next
  
                resumeABCSMC!(tracker, threshold, input.n_particles, input.file_path;
                    write_progress = write_progress,
                    progress_every = progress_every)

                if !tracker.can_continue
                    break
                end

                go_to_next = true
                

            else 
                
                iterateABCSMC!(tracker, threshold, input.n_particles, input.file_path;
                    write_progress = write_progress,
                    progress_every = progress_every)

                if !tracker.can_continue
                    break
                end

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
    else
        @warn "No particles selected at initial rejection ABC step of simulated SMC ABC - terminating algorithm"
    end

    return buildAbcSmcOutput(tracker), q_history
end
