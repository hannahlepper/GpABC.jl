#function for continuing a simulation when given necessary components
#Continue: threshold schedule
#First need to reconstruct the fit object, either from an existing object or from the components of one

#method one - take an existing fit object
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
    #idea is that i want to either take the whole fitted object and contiue where i left off

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

#method two - takes previous outputs in separate fields
function SimulatedABCSMC_continue(
    populations::AbstractArray{AbstractArray{Float64,2},1},
    distances::AbstractArray{AbstractArray{Float64,1},1},
    weights::AbstractArray{StatsBase.Weights,1},
    n_tries::AbstractArray{Int64,1},
    n_accepted::AbstractArray{Int64,1},
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
    #idea is that i want to either take the whole fitted object and contiue where i left off

    n_params = length(priors)

    summary_statistic = build_summary_statistic(summary_statistic)
    reference_summary_statistic = summary_statistic(reference_data)
    distance_simulation_input = DistanceSimulationInput(reference_summary_statistic, simulator_function, summary_statistic, distance_function)
    input = SimulatedABCSMCInput(n_params, n_particles, threshold_schedule,
                                    priors, distance_simulation_input,
                                    max_iter, file_path)

    n_pops_so_far = length(n_tries)

    tracker = SimulatedABCSMCTracker(input.n_params,
                            n_accepted,
                            n_tries,
                            input.threshold_schedule[1:n_pops_so_far],
                            n_accepted[end] > 0,
                            populations,
                            distances,
                            weights,
                            input.priors,
                            input.distance_simulation_input,
                            input.max_iter,
                            n_pops_so_far)


    return ABCSMC_continue(tracker,input; kwargs...)

end


function ABCSMC_continue(
        tracker::SimulatedABCSMCTracker,
        input::SimulatedABCSMCInput;
        write_progress = true,
        progress_every = 1000,
        ) 
    n_pops_so_far = tracker.pop_n
    if tracker.can_continue
        for i in n_pops_so_far+1:length(input.threshold_schedule)
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