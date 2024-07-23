##############
# Simulation #
##############
import numpy as np
import copy
import sem
import matplotlib.pyplot as plt

def adapt_dictionary_for_sims(params, step=4, simulationTime=1, dataRate=100):
    validation_params_sim = copy.deepcopy(params)
    validation_params_sim['dlFlowDataRate'] = [dataRate]
    validation_params_sim['ulFlowDataRate'] = [dataRate]
    validation_params_sim['simulationTime'] = [simulationTime]
    params['nEhtStations'] = [0]
    validation_params_sim['nStations'] = list(range(params['nStations'][0], params['nStations'][-1]+1, step))
    return validation_params_sim


def adapt_dictionary_for_sims_improved(params, step=4, simulationTime=1, dataRate=100, additional_params=None):
    """
    Modify the given parameters dictionary for simulation purposes. It adapts certain key values 
    and can incorporate additional simulation parameters.

    Args:
    params (dict): The original parameters dictionary.
    dataRate (int): Data rate for both 'dlFlowDataRate' and 'ulFlowDataRate'. Default is 100.
    additional_params (dict): Additional parameters to be included or modified in the dictionary. Default is None.

    Returns:
    dict: A modified dictionary adapted for simulations.
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary.")

    validation_params_sim = copy.deepcopy(params)
    validation_params_sim['dlFlowDataRate'] = [dataRate]
    validation_params_sim['ulFlowDataRate'] = [dataRate]
    validation_params_sim['simulationTime'] = [simulationTime]

    if 'nStations' in params:
        params['nEhtStations'] = [0]
        validation_params_sim['nStations'] = list(range(params['nStations'][0], params['nStations'][-1] + 1, step))
    else:
        raise KeyError("'nStations' key not found in params.")

    if additional_params and isinstance(additional_params, dict):
        validation_params_sim.update(additional_params)

    return validation_params_sim

def plot_with_cis(results, runs, x, multiple_lines=False):
    results = results.squeeze()
    std = np.transpose(results.reduce(np.std, 'runs').data)
    avg = np.transpose(results.reduce(np.mean, 'runs').data)
    # ci = 1.96 * std / np.sqrt(runs)
    # ci = 2.32 * std / np.sqrt(runs)
    ci = 2.576 * std / np.sqrt(runs)
    # print(avg)
    # avg.plot.line('-', x=x)
    medianprops = dict(linestyle='None')
    boxprops = dict(linestyle='None')
    whiskerprops = dict(linewidth=1)
    ax = plt.gca()
    if len(avg.shape) > 1:
        for i in range(len(avg)):
            # p = ax.plot(results.coords[x].data, avg[i], ':')
            for x_idx, x_value in enumerate(results.coords[x].data):
                item = {}
                item["med"] = avg[i][x_idx]
                item["q1"] = avg[i][x_idx]
                item["q3"] = avg[i][x_idx]
                item["whislo"] = avg[i][x_idx] - ci[i][x_idx]
                item["whishi"] = avg[i][x_idx] + ci[i][x_idx]
                stats = [item]
                ax.bxp(stats,
                       positions=[x_value],
                       showfliers=False,
                       medianprops=medianprops,
                       boxprops=boxprops,
                       whiskerprops=whiskerprops)
                # ax.fill_between(results.coords[x].data, (avg[i]-ci[i]), (avg[i]+ci[i]), color=p[0].get_color(), alpha=.1)
    else:
        # p = ax.plot(results.coords[x].data, avg, ':')
        for x_idx, x_value in enumerate(results.coords[x].data):
            item = {}
            item["med"] = avg[x_idx]
            item["q1"] = avg[x_idx]
            item["q3"] = avg[x_idx]
            item["whislo"] = avg[x_idx] - ci[x_idx]
            item["whishi"] = avg[x_idx] + ci[x_idx]
            stats = [item]
            ax.bxp(stats,
                   positions=[x_value],
                   showfliers=False,
                   medianprops=medianprops,
                   boxprops=boxprops,
                   whiskerprops=whiskerprops)


def get_simulation_campaign(validation_params, runs, overwrite):
    campaign = sem.CampaignManager.new('../../../..', 'wifi-ofdma-validation',
                                       'ofdma-validation-results',
                                       runner_type='ParallelRunner',
                                       overwrite=overwrite,
                                       optimized=True,
                                       check_repo=False,
                                       max_parallel_processes=1)
    params = copy.deepcopy(validation_params)
    params['verbose'] = [False]
    # params['printToFile'] = [True]
    campaign.run_missing_simulations(params, runs)
    return campaign


def get_simulation_campaign_improved(validation_params, runs, overwrite, simulation_dir='../../../..', 
                                     campaign_name='wifi-ofdma-validation', results_folder='ofdma-validation-results',
                                     additional_params=None):
    
    campaign = sem.CampaignManager.new(simulation_dir, campaign_name, results_folder,
                                       runner_type='ParallelRunner', overwrite=overwrite,
                                       optimized=True, check_repo=False, max_parallel_processes=1)

    params = copy.deepcopy(validation_params)
    params['verbose'] = [False]

    if additional_params and isinstance(additional_params, dict):
        params.update(additional_params)

    campaign.run_missing_simulations(params, runs)

    return campaign


def get_metrics(result):
    lines = iter(result['output']['stdout'].splitlines())
    dl = ul = dllegacy = ullegacy = hol = dlmucomp = hetbcomp = 0
    # dls = uls = dlslegacy = ulslegacy = []
    dls = uls = []
    while (True):
        line = next(lines, None)
        if line is None:
            break
        if "Per-AC" in line:
            line = next(lines, None)
            line = next(lines, None)
            dls += [float(line.split(" ")[-1])]
            line = next(lines, None)
            line = next(lines, None)
            uls += [float(line.split(" ")[-1]) if line.split(" ")[-1] else 0]
            # line = next(lines, None)
            # line = next(lines, None)
            # dlslegacy += [float(line.split(" ")[-1])]
            # line = next(lines, None)
            # line = next(lines, None)
            # ulslegacy += [float(line.split(" ")[-1])]
        if "Throughput (Mbps) [DL]" in line:
            # Go down to total
            while (True):
                line = next(lines, None)
                if line is None:
                    break
                if "TOTAL:" in line:
                    dl = float(line.split(" ")[-1])
                    break
        if "Throughput (Mbps) [UL]" in line:
            # Go down to total
            while (True):
                line = next(lines, None)
                if line is None:
                    break
                if "TOTAL:" in line:
                    ul = float(line.split(" ")[-1])
                    break
        if "Throughput (Mbps) [DL] LEGACY" in line:
            # Go down to total
            while (True):
                line = next(lines, None)
                if line is None:
                    break
                if "TOTAL:" in line:
                    dllegacy = float(line.split(" ")[-1])
                    break
        if "Throughput (Mbps) [UL] LEGACY" in line:
            # Go down to total
            while (True):
                line = next(lines, None)
                if line is None:
                    break
                if "TOTAL:" in line:
                    ullegacy = float(line.split(" ")[-1])
                    break
        if "Pairwise Head-of-Line delay" in line:
            # Go down to total
            while (True):
                line = next(lines, None)
                if line is None:
                    break
                if "TOTAL:" in line:
                    count = float(line.split("[")[1].split("]")[0])
                    if (count == 0):
                        hol = 0
                        break
                    hol = float(line.split("<")[1].split(">")[0])
                    break
        if "DL MU PPDU completeness" in line:
            # Go down to total
            line = next(lines, None)
            line = next(lines, None)
            dlmucomp = 0
            break
        if "HE TB PPDU completeness" in line:
            # Go down to total
            line = next(next(lines, None))
            line = next(lines, None)
            hetbcomp = 0
            break
    return [dl, ul, dllegacy, ullegacy, hol, dlmucomp, hetbcomp]

def print_detailed_simulation_output (validation_params, overwrite=False):

    campaign = sem.CampaignManager.new('../../../../', 'wifi-ofdma-validation',
                                       'validation-results',
                                       runner_type='ParallelRunner',
                                       overwrite=overwrite,
                                       check_repo=False)
    params = copy.deepcopy(validation_params)
    params['verbose'] = [True]
    # params['printToFile'] = [True]
    campaign.run_missing_simulations(params)

    example_result = next(campaign.db.get_complete_results(params))

    print(example_result['output']['stdout'])

    print(sem.utils.get_command_from_result('wifi-ofdma-validation',
                                            example_result))

    for line in example_result['output']['stdout'].splitlines():
        if "Starting statistics" in line:
            start = int(line.split(" ")[-1])
        if "Stopping statistics" in line:
            stop = int(line.split(" ")[-1])

    print("Transmission times for devices: ")
    for line in example_result['output']['WifiPhyStateLog.txt'].splitlines():
        cur_time, context, time, duration, state = line.split(" ")
        if state == "TX" and float(time) > start and float(time) < stop:
            print("Time %s: device %s transmitted for %s" % (time, context, duration))

    print("Simulation throughput: %s" % get_metrics(example_result))
    return example_result


def get_simulation_metrics(validation_params, runs=2, overwrite=False, results_folder='validation-results', verbose=False):

    campaign = sem.CampaignManager.new('../../../../', 'wifi-ofdma-validation',
                                       results_folder,
                                       runner_type='ParallelRunner',
                                       overwrite=overwrite,
                                       check_repo=False,
                                       max_parallel_processes=1)

    params = copy.deepcopy(validation_params)
    params['verbose'] = [False]
    if runs == None:
        campaign.run_missing_simulations(params)
    else:
        campaign.run_missing_simulations(params, runs)

    if verbose:
        for result in campaign.db.get_results(params):
            print(sem.utils.get_command_from_result('wifi-ofdma-validation', result))
    throughput_results = campaign.get_results_as_xarray(
        params, get_metrics, ['dl', 'ul', 'dllegacy', 'ullegacy', 'hol',
                              'dlmucomp', 'hetbcomp'], runs).squeeze()

    return throughput_results

def get_metrics_improved(result):
   
    try:
        lines = iter(result['output']['stdout'].splitlines())
        dl = ul = dllegacy = ullegacy = hol = dlmucomp = hetbcomp = 0
        dls = uls = []

        while True:
            line = next(lines, None)
            if line is None:
                break

            if "Per-AC" in line:
                line = next(lines, None)  # Skip two lines to reach the relevant data
                line = next(lines, None)
                dls.append(float(line.split()[-1]))
                line = next(lines, None)  # Skip two lines for UL data
                line = next(lines, None)
                ul_value = line.split()[-1]
                uls.append(float(ul_value) if ul_value else 0)

            # Parsing other metrics
            if "Throughput (Mbps) [DL]" in line:
                dl = get_total_metric_value(lines)
            elif "Throughput (Mbps) [UL]" in line:
                ul = get_total_metric_value(lines)
            elif "Throughput (Mbps) [DL] LEGACY" in line:
                dllegacy = get_total_metric_value(lines)
            elif "Throughput (Mbps) [UL] LEGACY" in line:
                ullegacy = get_total_metric_value(lines)
            elif "Pairwise Head-of-Line delay" in line:
                hol = get_hol_metric_value(lines)
            elif "DL MU PPDU completeness" in line or "HE TB PPDU completeness" in line:
                # Set to 0 or a specific value if needed
                dlmucomp = hetbcomp = 0

        return [dl, ul, dllegacy, ullegacy, hol, dlmucomp, hetbcomp]

    except Exception as e:
        print(f"Error processing metrics: {e}")
        return [0, 0, 0, 0, 0, 0, 0]

def get_total_metric_value(lines):
    """
    Helper function to extract the 'TOTAL:' metric value from a line iterator.
    """
    while True:
        line = next(lines, None)
        if line is None or "TOTAL:" in line:
            return float(line.split()[-1]) if line else 0

def get_hol_metric_value(lines):
    """
    Helper function to extract the 'Pairwise Head-of-Line delay' metric value from a line iterator.
    """
    while True:
        line = next(lines, None)
        if line is None:
            return 0
        if "TOTAL:" in line:
            count = float(line.split("[")[1].split("]")[0])
            return float(line.split("<")[1].split(">")[0]) if count != 0 else 0

def get_simulation_metrics_improved(validation_params, runs=2, overwrite=False, results_folder='validation-results', 
                                    verbose=False, additional_metrics=None, custom_campaign_manager=None):
  
    default_metrics = ['dl', 'ul', 'dllegacy', 'ullegacy', 'hol', 'dlmucomp', 'hetbcomp']
    if additional_metrics is not None:
        if not isinstance(additional_metrics, list):
            raise ValueError("additional_metrics must be a list.")
        metrics = default_metrics + additional_metrics
    else:
        metrics = default_metrics

    if custom_campaign_manager:
        campaign = custom_campaign_manager
    else:
        campaign = sem.CampaignManager.new('../../../../', 'wifi-ofdma-validation', results_folder,
                                           runner_type='ParallelRunner', overwrite=overwrite,
                                           check_repo=False, max_parallel_processes=1)

    params = copy.deepcopy(validation_params)
    params['verbose'] = [verbose]

    if runs is None:
        campaign.run_missing_simulations(params)
    else:
        campaign.run_missing_simulations(params, runs)

    if verbose:
        for result in campaign.db.get_results(params):
            print(sem.utils.get_command_from_result('wifi-ofdma-validation', result))

    throughput_results = campaign.get_results_as_xarray(
        params, get_metrics, metrics, runs).squeeze()

    return throughput_results
