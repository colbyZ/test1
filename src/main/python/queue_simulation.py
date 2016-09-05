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
