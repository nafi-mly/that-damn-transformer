### What I've learned from this video
1. we don't set a lot of derivatives cases, but rather using the definition of derivative to derive thing from scratch. It's like a general one size-fit-all function. It's about how much change when we nudge a variable at a moment. It's also about h is normalize/descale it. This rewire my brain bro, really
2. just found out there's an issue on an extremely small floating point
3.     some special methods:
        __add__     → triggered by +
        __mul__     → triggered by *
        __repr__    → triggered by repr(), print(), (fallback), ...
        __init__    → triggered on object construction
        __sub__     → triggered by -
        __eq__      → triggered by ==
        __len__     → triggered by len()
        __getitem__ → triggered by ([]), 
        __call__    → triggered by (()), etc.
4. that Value class is extremely useful when we have a complex data structure, say like n-dimension vector
5. casually crashed out because i didnt know how the fck venv actually work (no, i am not talking about that basic venv, i am talking about the engine behind it)
6. about derivaties, this means that we just need to do two multiplication for each node because we got the partial derivation from the n+1 laeyer, so... to backprop for the entire NN, it's ~2 x params. If we do this for N of time, it will be... oh shoot. But i know that we aint calculate the whole things.
7. this really changes me the way i look on sum and product bro, really. It plays the different game when we talk about sensitivity