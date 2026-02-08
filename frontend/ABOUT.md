## team members
our party consists of toby cox, ryon peddapalli, and jacob davis!

## project purpose
inspired by the sam & cat magic atm skit, we cast offkey, a tool for converting audio into replicable passwords and hashes for authentication and security sturdier than the knock spell! popular tunes and songs can be easier to remember than strings of passwords, and offkey provides a solution to help prevent that pesky annoyance of forgetting your password. offkey is valuable for authenticating voices over phone calls for important transactions for example, and helps assist elder community members and those with poor memory use their passwords, as music and sound are easy to recall.

## how we built it
our backend is written in python and we use django for our frontend from javascript, html, and css code. we make use of the scipy, numpy, and librosa libraries in python, and offkey.tech is our domain name (pending registration). we have some tests using speechbrain but offkey ultimately does not use it.

## challenges we overcame
determining a solid metric and method for reliably classifying the audio took up most of our time. factors that made this challenging include accounting for the time domain, understanding how to determine a voice different from another, and accounting for minute differences that could result in an entirely different hash. we explored methods including the fast fourier transform, mel-frequency cepstral coefficients, and deep learning models with pytorch. we landed on a traditional method using the fast fourier transform for its relative computational efficiency and scalability, conducting a parameter space search to find an optimal configuration for determining suitable audio clips. we also sample input audio in discrete timesteps across a fixed number of buckets for increased performance and more consistent classifications.
