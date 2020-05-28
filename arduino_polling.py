import numpy as np


def capture_sensvalues(ard, no_sensors=5, data_points=64, baseline_remove=True):
    """
    Returns numpy array of sensor values taken from arduino through the serial port
    :param data_points:
    :param baseline_remove:
    :param ard:
    :param no_sensors: number of sensors the arduino has
    :return:
    """
    ard.write(b'c')
    values = ard.readlines()
    # print(values)
    data = np.zeros([data_points, no_sensors])
    times = np.zeros(data_points)
    for i in range(data_points):
        vals = values[i * (no_sensors + 1):(i + 1) * (no_sensors + 1)]
        for j in range(no_sensors):
            data[i, j] = vals[j + 1][:-2]
        times[i] = vals[0][:-2]
    times = times * 1e-6
    if baseline_remove:
        for i in range(no_sensors):
            data[:, i] = data[:, i] / np.int(np.round(np.mean(data[:, i])))
    return times, data


def quickcapture_sensvalues(ard, no_sensors=5, data_points=64, baseline_remove=True):
    """
    Returns numpy array of sensor values taken from arduino through the serial port
    :param ard:
    :param no_sensors: number of sensors the arduino has
    :return:
    """
    ard.write(b'c')
    values = ard.readlines()
    # print(values)
    data = np.zeros([data_points, no_sensors])
    for i in range(data_points):
        vals = values[i * no_sensors:(i + 1) * no_sensors]
        for j in range(no_sensors):
            data[i, j] = vals[j][:-2]
    if baseline_remove:
        for i in range(no_sensors):
            data[:, i] = data[:, i] / np.int(np.round(np.mean(data[:, i])))
    return [0], data


def get_sensamp(ard):
    t, data = quickcapture_sensvalues(ard, baseline_remove=False)
    data = np.max(data, 0) - np.min(data, 0)
    return np.abs(data)


def get_sensmag(ard, baseline=[]):
    t, data = quickcapture_sensvalues(ard, baseline_remove=False)
    avgs = np.mean(data, 0)
    if len(baseline) < 4:
        return avgs
    else:
        return np.abs(avgs - baseline)


def get_sensfourier(times, data):
    """
    returns the fourier spectrum for the sensor data
    :param times:
    :param data:
    :return:
    """
    from scipy import interpolate
    signals = data.shape[1]
    no_points = 256
    d = np.zeros([no_points, signals])
    t_new = np.linspace(times[0], times[-1], no_points)

    for i in range(signals):
        d_b = data[:, i]
        f = interpolate.interp1d(times, d_b)
        d_new = f(t_new)
        d[:, i] = d_new
        # plt.plot(t_new,d_new)
        # plt.plot(times, d_b)
        # plt.show()

    length_t = t_new[-1] - times[0]

    fs = no_points / length_t
    f = np.linspace(0, 1, -1 + no_points / 2) * fs / 2
    specs = np.zeros([no_points / 2 - 1, signals])

    for i in range(signals):
        Y = np.fft.fft(d[:, i], no_points) / no_points
        specs[:, i] = 2 * abs(Y[0:no_points / 2 - 1])
    return f, specs
