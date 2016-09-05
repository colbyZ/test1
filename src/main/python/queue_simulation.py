import numpy as np

arrival_gaps = np.random.uniform(1.0, 20.0, 100)
print 'arrival gaps:\n%s\n' % arrival_gaps

arrival_time = 0.0
arrival_times = []
for gap in arrival_gaps:
    arrival_time += gap
    arrival_times.append(arrival_time)

print 'arrival times: \n%s\n' % arrival_times

service_times = np.random.uniform(5.0, 15.0, 100)
print 'service times:\n%s\n' % service_times

from collections import deque


def simulation(arrival_times, service_times):
    queue = deque()
    wait_times = []

    for i in range(len(arrival_gaps)):
        arrival_time = arrival_times[i]

        while queue and queue[0] <= arrival_time:
            if queue[0] <= arrival_time:
                queue.popleft()

        service_time = service_times[i]

        if queue:
            service_start_time = queue[-1]
        else:
            service_start_time = arrival_time

        service_finish_time = service_start_time + service_time
        queue.append(service_finish_time)
        wait_time = service_start_time - arrival_time
        wait_times.append(wait_time)

    avg_wait_time = np.mean(wait_times)
    print avg_wait_time

    return 0, avg_wait_time


simulation(arrival_times, service_times)
