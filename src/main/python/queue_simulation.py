import numpy as np

from collections import deque


def simulation(arrival_times, service_times):
    queue = deque()
    wait_times = []

    queue_length_by_time_sum = 0.0

    prev_time = 0.0

    for i in range(len(arrival_times)):
        arrival_time = arrival_times[i]

        while queue and queue[0] <= arrival_time:
            queue_length_by_time_sum += len(queue) * (queue[0] - prev_time)
            prev_time = queue.popleft()

        service_time = service_times[i]

        if queue:
            service_start_time = queue[-1]
        else:
            service_start_time = arrival_time

        service_finish_time = service_start_time + service_time

        queue_length_by_time_sum += len(queue) * (arrival_time - prev_time)
        queue.append(service_finish_time)
        prev_time = arrival_time

        wait_time = service_start_time - arrival_time
        wait_times.append(wait_time)

    while queue:
        queue_length_by_time_sum += len(queue) * (queue[0] - prev_time)
        prev_time = queue.popleft()

    avg_wait_time = np.mean(wait_times)
    avg_queue_length = queue_length_by_time_sum / prev_time

    return avg_queue_length, avg_wait_time


def get_arrival_and_service_times():
    arrival_gaps = np.random.uniform(1.0, 20.0, 100)

    arrival_time = 0.0
    arrival_times = []
    for gap in arrival_gaps:
        arrival_time += gap
        arrival_times.append(arrival_time)

    service_times = np.random.uniform(5.0, 15.0, 100)

    return arrival_times, service_times


avg_wait_times = []
avg_queue_lengths = []

for i in range(500):
    arrival_times, service_times = get_arrival_and_service_times()
    avg_queue_length, avg_wait_time = simulation(arrival_times, service_times)

    avg_wait_times.append(avg_wait_time)
    avg_queue_lengths.append(avg_queue_length)

print 'avg wait times, mean: %.2f, std: %.2f' % (np.mean(avg_wait_times), np.std(avg_wait_times))
print 'avg queue lengths, mean: %.2f, std: %.2f' % (np.mean(avg_queue_lengths), np.std(avg_queue_lengths))

print avg_wait_times
