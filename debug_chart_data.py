import numpy as np

# Simulate what the hook does:
# Backend sends: [{'heart_rate': 10}, {'heart_rate': 11}, ...] (600 points)
# Hook transforms this into: [{x: ts1, y: 10}, {x: ts2, y: 11}, ...]
# This is what SystemCluster passes to DualScaleChart.

# The flaw likely lies in the chart.js 'data' slice logic:
# .slice(-(duration + 1))

# If duration=600 and the points array has 600 items:
# points.slice(-601) -> returns all 600 items. Correct.

# Wait, check the DualScaleChart code:
# points.slice(-(duration + 1)).map(...)

# What if 'points' is just initialized to []?
# Points: []
# slice(-601) -> []
# map -> []
# Line chart gets data: [] -> Empty graph.
