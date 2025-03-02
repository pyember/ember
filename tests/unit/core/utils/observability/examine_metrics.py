#!/usr/bin/env python
"""
Script to examine metrics objects structure for testing.
"""

import pprint
import inspect
from prometheus_client import Counter, Histogram, CollectorRegistry

def examine_metric_objects():
    """Examine the structure of Prometheus metric objects."""
    
    # Create a registry
    registry = CollectorRegistry()
    
    # Create metrics
    counter = Counter('test_counter', 'Test counter', ['label'], registry=registry)
    histogram = Histogram('test_histogram', 'Test histogram', ['label'], registry=registry)
    
    # Add some data
    counter.labels(label='test').inc()
    histogram.labels(label='test').observe(0.5)
    
    # Examine the counter
    print("\n=== COUNTER EXAMINATION ===")
    
    label_values = ('test',)
    counter_metric = counter._metrics.get(label_values)
    
    print(f"Counter metric object type: {type(counter_metric)}")
    print(f"Counter metric object dir: {dir(counter_metric)}")
    
    if hasattr(counter_metric, 'get'):
        print(f"Counter metric get() result: {counter_metric.get()}")
    
    if hasattr(counter_metric, '_value'):
        print(f"Counter metric _value: {counter_metric._value}")
    
    # Examine the histogram
    print("\n=== HISTOGRAM EXAMINATION ===")
    
    histogram_metric = histogram._metrics.get(label_values)
    
    print(f"Histogram metric object type: {type(histogram_metric)}")
    print(f"Histogram metric object dir: {dir(histogram_metric)}")
    
    # Try different ways to get count and sum
    count_attributes = ['_count', 'count', 'get_sample_count']
    for attr in count_attributes:
        if hasattr(histogram_metric, attr):
            value = getattr(histogram_metric, attr)
            if callable(value):
                value = value()
            print(f"Histogram count via {attr}: {value}")
    
    sum_attributes = ['_sum', 'sum', 'get_sample_sum']
    for attr in sum_attributes:
        if hasattr(histogram_metric, attr):
            value = getattr(histogram_metric, attr)
            if callable(value):
                value = value()
            print(f"Histogram sum via {attr}: {value}")

if __name__ == "__main__":
    examine_metric_objects()