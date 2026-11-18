# Section 2 — Exploratory Data Analysis and Data Quality

## Scope

This analysis covers the 80-trip training partition and the 10-trip student-test partition. The reserved `evaluator_test` partition was not opened. The EDA implementation is in `src/lane_a_data/eda_stats.py`, `src/lane_a_data/eda_plots.py`, and `src/lane_a_data/run_eda.py`.

## Dataset inventory

Each trip contains the expected `.track`, `.nodes`, `.arcs`, and `.route` files. Across the partitions, all checked trip files were present, and the EDA found no NaN values, invalid coordinates, unknown node or route references, repeated timestamps, or backwards timestamps.

| Measure | Training | Student test |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| GPS observations per trip (median) | 1,529 | 1,420 |
| Median network nodes per trip | 9,608 | 6,597 |
| Median network arcs per trip | 19,527 | 13,145 |
| Recorded time | 126.2 h | 5.6 h |

The largest network contains 307,392 arcs. The trip networks and GPS observations cover 25 UTM zones in total (longitude −100.4° to 137.2°, latitude −34.6° to 61.7°).

## Sampling and time gaps

Sampling is close to 1 Hz for most observations: 96.5% of training intervals and 98.6% of student-test intervals are one second. There are 1,216 training intervals and 44 student-test intervals longer than five seconds. The longest gaps are 153,064 seconds in training and 210 seconds in student testing. Non-integer timestamps occur in three training trips (9,659 values); none were reported in student testing.

These findings support treating long gaps explicitly. Preprocessing flags gaps and segments the track so smoothing and heading calculations do not cross interruptions.

## Geographic projection

The data spans many regions, so a single local projection would not suit all trips. Each trip is projected into its own UTM zone, with its track and road network in the same CRS. This keeps distance and geometry calculations in meters. Web Mercator was not used for these measurements because its scale distortion is substantial at the dataset’s latitudes.

## GPS noise, speed, and heading

Raw GPS points show short-scale jitter. The median share of training step pairs turning by more than 150° is 6%; 18 training trips exceed 15%. Across the training partition, raw GPS path length totals 2,968 km. After smoothing it totals 2,261 km, close to the 2,229 km reference-route total.

The heading study found that headings from consecutive raw points point more than 90° away from the nearest reference-arc direction for 16.8% of training points, with a median error of 23.6°. The selected method—a 9-point position smoothing window and a ±2-point heading window—reduces these to 6.1% and 3.3°. Median moving speed after smoothing is 28 km/h in training and 44 km/h in student testing.

Raw speed jumps above 252 km/h were flagged in 10 training trips (150 jumps total) and one student trip (one jump). These values are flagged as outliers rather than silently removed. Consecutive identical GPS positions were rare: six in five training trips and none in student testing.

## Road-network and reference-route checks

The median share of road arcs with a reverse-direction twin is 95% in training. Direction therefore matters: opposite-direction arcs may share geometry, and heading is useful for distinguishing them. The median share of used nodes in the largest strongly connected component is 96% in training.

The EDA also flagged network defects: 1,775 duplicate arcs in 53 training trips and 128 in seven student trips; 100 near-zero-length arcs in 29 training trips and six in three student trips; and 22 self-loop arcs in eight training trips and two in one student trip.

Reference routes contain transitions between unconnected directed arcs in 10 training trips (13 transitions total) and one student trip (one transition). GPS-to-nearest-reference-route distance has a training median of 5.6 m and 95th percentile of 140.3 m; the student-test median is 5.2 m and 95th percentile is 16.8 m.

There are 22,576 training points more than 50 m from their nearest reference-route arc. Of those, 4,345 lie on another supplied road, suggesting that the reference route may omit part of the drive. Another 18,231 training points are more than 50 m from any supplied road. In student testing, 27 points are over 50 m from the reference route, and all 27 lie on another road; none are over 50 m from every road in the network.

These discrepancies matter for reference labels. Forcing a distant point onto the nearest route arc can produce an incorrect label. Far-off points should remain unlabeled, and trips with widespread off-network points or broken route transitions should be reviewed before training.

## Preprocessing decisions supported by EDA

- Preserve raw coordinates and every observation row; `obs_idx` remains the join key.
- Project each trip’s observations and network into that trip’s UTM zone.
- Flag large time gaps, reversed time, speed outliers, and stationary points instead of dropping them.
- Smooth positions with a 9-point rolling window and calculate heading using a ±2-point central difference.
- Do not smooth across gaps longer than five seconds.
- Require sufficient movement before accepting a new heading; otherwise carry the prior heading and mark the value invalid.
- Keep unreliable or unavailable reference labels unlabeled rather than converting them to negatives.

## Limitations and interpretation

The partitions are structurally complete, but quality varies by trip. A small number of long gaps, noisy trajectories, route/network discrepancies, and disconnected route transitions can affect feature quality and ground-truth alignment. Summary statistics should therefore be presented together with the per-trip anomaly inventory; aggregate medians alone hide problematic trips such as long-gap or off-network cases.

The numerical findings above were generated by `run_eda.py` over `train` and `student_test`. EDA tables and figures were written to a temporary directory during this run; the report contains the relevant aggregate results so it can be read independently.
