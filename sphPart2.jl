# NYU CS 421 Numerical Computing SPH project
# Prepared by Dr. Gizem Kayar in April 2026
# To be completed by Spring 2026 CS 421 students

using LinearAlgebra
using CSV
using DataFrames
using Makie

# -----------------------------
# Parameters
# -----------------------------

 # TO DO 6
const h    = 0.04 
const mass = 1.06
const rho0 = 1000.0
const k    = 3000.0
const mu   = 0.1
const g    = [0.0, -9.8]
const dt   = 2.0e-5
const STEPS = 100000
const SAVE_EVERY = 10
const dx   = h * 0.8

const cell_size = h
const grid_res = Int(ceil(1.0 / cell_size))  # domain [0,1]

# -----------------------------
# Kernels
# -----------------------------

 # TO DO 7
function W_cubic(r)
    q = r / h
    s = 3 / (2 *pi*h^2)

    if 0 <= q < 1
        return s * (2/3 - q^2 + 0.5*q^3)

    elseif 1 <= q < 2
       return s * (1/6* (2 - q)^3)

    else
       return 0.0
    end
end

function gradW_spiky(rvec)
    r = norm(rvec)
    (0 < r ≤ h) ? -30/(pi*h^5)*(h - r)^2*(rvec/r) : zeros(2)
end

lapW_visc(r) = (0 ≤ r ≤ h) ? 20/(3*pi*h^5)*(h - r) : 0.0

# -----------------------------
# Initialization (corner dam)
# -----------------------------
function init_particles()
    
    pos = Vector{Vector{Float64}}()
    vel = Vector{Vector{Float64}}()

    # Dense block in lower-left corner
    # TO DO 1 and 8
    for  x = dx:dx:0.4
        for y = dx:dx:0.6
            push!(pos, [x, y])
            push!(vel, [0.0, 0.0])
        end

    end 

    Np = length(pos)
    rho = zeros(Np)
    P   = zeros(Np)

    return pos, vel, rho, P
end



# -----------------------------
# Neighbor search
# -----------------------------
function find_neighbors(pos)

    neighbors = [Int[] for _ in eachindex(pos)]

    n = length(pos)

    for i in 1:n
        for j in 1:n
            if i != j

                r = norm(pos[i] - pos[j])

                if r < 2h
                    push!(neighbors[i], j)
                end
            end
        end
    end

    return neighbors
end

# -----------------------------
# Density & Pressure
# -----------------------------
function compute_density!(pos, rho, neighbors)
    for i in eachindex(pos)
        ρ = mass * W_cubic(0.0) #initialize density with self contribution

        # TO DO 3 - compute density via smoothing

        for j in neighbors[i]
            r = norm(pos[i] - pos[j])
            ρ += mass * W_cubic(r)
        end
        rho[i] = max(ρ, 1e-6)  # prevent division issues
    end
end

compute_pressure!(rho, P) = (P .= max.(k .* (rho .- rho0), 0.0))

# -----------------------------
# Forces
# -----------------------------
function compute_forces(pos, vel, rho, P, neighbors)
    forces = [zeros(2) for _ in eachindex(pos)]

    for i in eachindex(pos)
        f_p = zeros(2)
        f_v = zeros(2)

        for j in neighbors[i]
            rij = pos[i] - pos[j]
            r = norm(rij)

            
             # TO DO 4 - compute pressure force and viscosity force

            # Symmetric pressure force (stable)
            f_p += -(mass * (P[i]/rho[i]^2 + P[j]/rho[j]^2) * gradW_spiky(rij))
           
            # Viscosity
            f_v += mu * mass * (vel[j] - vel[i]) / rho[j] * lapW_visc(r)
        end

        forces[i] = f_p + f_v + g #gravity also added
    end

    return forces
end

# -----------------------------
# Integration (Euler)
# -----------------------------
function integrate!(pos, vel, accel)

    for i in eachindex(pos)

        vold = copy(vel[i])

        pos[i] += dt * vold
        vel[i] += dt * accel[i]

        # boundary: left and right walls (reflective, damped)
        if pos[i][1] < 0.0
            pos[i][1] = 0.0
            vel[i][1] = abs(vel[i][1]) * 0.5

        elseif pos[i][1] > 1.0
            pos[i][1] = 1.0
            vel[i][1] = -abs(vel[i][1]) * 0.5
        end

        # boundary: bottom wall (reflective, more damped)
        if pos[i][2] < 0.0
            pos[i][2] = 0.0
            vel[i][2] = abs(vel[i][2]) * 0.3
        end
         # top is open
    end
end

function xsph!(vel, pos, rho, neighbors)
    ε = 0.5
    newvel = deepcopy(vel)

    for i in eachindex(pos)
        corr = zeros(2)
        for j in neighbors[i]
            corr += mass * (vel[j] - vel[i]) / rho[j] *
                    W_cubic(norm(pos[i] - pos[j]))
        end
        newvel[i] += ε * corr
    end

    for i in eachindex(vel)
        vel[i] = newvel[i]
    end
end

# -----------------------------
# CSV output
# -----------------------------
function save_csv(pos, vel, rho, P, step)
    outdir = "output"
    isdir(outdir) || mkdir(outdir)

    filename = joinpath(outdir, "sph_$(lpad(step,6,'0')).csv")

    open(filename, "w") do io
        for p in pos
            println(io, "$(p[1]),$(p[2])")
        end
    end
end

# -----------------------------
# Main loop
# -----------------------------
function run()
    pos, vel, rho, P = init_particles()

    for step in 1:STEPS
        neighbors = find_neighbors(pos)

        compute_density!(pos, rho, neighbors)
        compute_pressure!(rho, P)
        accel = compute_forces(pos, vel, rho, P, neighbors)
        integrate!(pos, vel, accel)

        if step % SAVE_EVERY == 0
            save_csv(pos, vel, rho, P, step)
        end
    end

    println("Done (corner breaking dam).")
end

run()