

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from depth import DepthEucl
from depth.model.multivariate import *


def interpolate_chunks(df, start_time, end_time, step_seconds=10, chunk_len_min=5):
    """
    Interpolate data in fixed-length chunks between start_time and end_time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex and numeric columns to interpolate.
    start_time, end_time : pd.Timestamp or str
        Start and end of the time window.
    step_seconds : int, default=10
        Step size of the interpolation grid inside each chunk.
    chunk_len_min : int, default=1
        Length of each chunk in minutes.

    Returns
    -------
    pd.DataFrame
        Interpolated values with index corresponding to the step grid for each chunk.
        Bins with <2 points are filled with NaN.
    """

    # Ensure datetime index
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Restrict data to the window
    df = df.loc[start_time:end_time]

    # List of chunk boundaries
    chunk_edges = pd.date_range(
        start_time.floor("min"),
        end_time.ceil("min"),
        freq=f"{chunk_len_min}min"
    )

    output_values = []
    output_times = []

    for t0 in chunk_edges:
        t1 = t0 + pd.Timedelta(minutes=chunk_len_min)

        # Extract chunk
        chunk = df[t0:t1]

        # Remove duplicate timestamps in the chunk
        chunk = chunk[~chunk.index.duplicated(keep='first')]

        # Step grid inside the chunk
        total_seconds = chunk_len_min * 60
        grid_seconds = np.arange(0, total_seconds + step_seconds, step_seconds)
        grid_times = t0 + pd.to_timedelta(grid_seconds, unit="s")

        # If <2 points → output NaNs
        if len(chunk) < 2:  ## we can check max time difference between two consecutive signals in this chunk and set max threshold
            output_values.append(
                np.full((len(grid_seconds), len(df.columns)), np.nan)
            )
            output_times.append(grid_times)
            continue

        # Convert timestamps to seconds from t0
        t = (chunk.index - t0).total_seconds().to_numpy()

        # Interpolate each column independently
        chunk_result = []
        for col in chunk.columns:
            y = chunk[col].to_numpy()
            valid = ~np.isnan(y)
            #print('time', t0, valid)
            if valid.sum() < 1:
                chunk_result.append(np.full(len(grid_seconds), np.nan))
                continue

            f = interp1d(
                t[valid], y[valid],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan
            )
            chunk_result.append(f(grid_seconds))

        # Stack columns
        chunk_result = np.column_stack(chunk_result)

        output_values.append(chunk_result)
        output_times.append(grid_times)

    # Combine all chunks
    all_times = np.concatenate(output_times)
    all_values = np.vstack(output_values)

    return pd.DataFrame(all_values, index=all_times, columns=df.columns)


###########################################################

def int_functional_depth(data, query_point, type_of_depth='halfspace', solver='neldermead', NRandom=100):
    """
    Compute the integrated functional depth (IFD) of a query function with respect to a sample of functional data.

    The function computes the average (integrated) multivariate depth of the query function
    across all discretization points. At each point, the multivariate depth is evaluated
    using the specified depth type and solver.

    Parameters
    ----------
    data : np.ndarray
        A 3D NumPy array of shape (N_data, L, D)
        representing the sample of functional data:
        - N_data: number of functional observations (samples)
        - L: number of discretization points per function
        - D: dimension of the data at each discretization point

        Each element data[j, i, :] corresponds to the D-dimensional
        observation of the j-th function at discretization point i.

    query_point : np.ndarray
        A 2D NumPy array of shape (L, D)
        representing the function for which the integrated depth is to be computed.
        It has the same structure as a single element of `data`.

    type_of_depth : str, optional, default='halfspace'
        The name of the multivariate depth function to use for each time slice.
        Typical choices include:
        - 'halfspace'
        - 'projection'
        - 'simplicial', etc.

    solver : str, optional, default='neldermead'
        The numerical method used to approximate the multivariate depth.
        For example, 'neldermead' uses the Nelder–Mead optimization algorithm.

    NRandom : int, optional, default=100
        The number of random projections (or random directions)
        used to approximate the depth when stochastic methods are involved.
        Larger values generally yield more accurate approximations at the cost of speed.

    Returns
    -------
    functional_depth_val : float
        The integrated functional depth of `query_point` with respect to `data`,
        computed as the average of the multivariate depths across all L discretization points.

    Notes
    -----
    For each discretization point i = 1, ..., L:
        - Extract the data slice `data[:, i, :]` (shape: N_data x D)
        - Extract the query vector `query_point[i, :]` (shape: D)
        - Compute the multivariate depth of the query vector relative to the data slice
        - Average the results over all L time points

    """
    total_depth_sum = 0
    l_points, d = query_point.shape

    for i in range(l_points):
        # data_component_slice: N_data x D matrix (all functions at time i)
        data_component_slice = data[:, i, :]

        # query_component: D-dimensional vector (query function at time i)
        query_component = query_point[i, :]

        # Compute depth at time i
        #time_component_depth = depth_approximation(
            #query_component, data_component_slice,
            #type_of_depth, solver, NRandom, option=1
        #)
        model=DepthEucl().load_dataset(data_component_slice)
        time_component_depth=model.halfspace(query_component,exact=False, output_option="lowest_depth")
        print("time component dept =", time_component_depth)
        #depth_approximation
        total_depth_sum += time_component_depth

    # Average depth over all L time points
    functional_depth_val = total_depth_sum / l_points

    return functional_depth_val


#################################################################


