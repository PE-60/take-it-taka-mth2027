"""ballq: the ball as a probability map over the pitch, from tracking data.

Modules, in the order the pipeline uses them:
    io          read and validate the raw files (the only module that opens a file)
    colours     shirt colours from the video: which team wears the light kit, mislabel flags
    features    per frame table: speeds, kicks, trust flags, dead ball windows
    grid        the pitch grid and distances
    motion      how the ball moves between frames (predict)
    observe     how much to believe each observation (update)
    filter      forward, backward, smooth, in chunks
    possession  Link and Hoernig on a point and on the posterior
    contract    output schemas and writers
    cli         python -m ballq run ...
"""
__version__ = "0.1.0"
