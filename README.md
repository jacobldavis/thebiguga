## Team Members
Our party consists of Toby Cox, Ryon Peddapalli, and Jacob Davis!

## Project purpose
Inspired by the Sam & Cat Magic ATM skit, we cast offkey, a tool for converting audio into replicable passwords and hashes for authentication and security sturdier than the knock spell! Popular tunes and songs can be easier to remember than strings of passwords, and offkey provides a solution to help prevent that pesky annoyance of forgetting your password. Offkey is valuable for authenticating voices over phone calls for important transactions for example, and helps assist elder community members and those with poor memory use their passwords, as music and sound are easy to recall.

## How we built it
Our backend is written in Python and we use Django for our frontend from JavaScript, HTML, and CSS code. We make use of the SciPy, NumPy, and Librosa libraries in Python, and offkey.tech is our domain name (pending registration).

## Challenges we overcame
Determining a solid metric and method for reliably classifying the audio took up most of our time. Factors that made this challenging include accounting for the time domain, understanding how to determine a voice different from another, and accounting for minute differences that could result in an entirely different hash. We explored methods including the Fast Fourier Transform, Mel-Frequency Cepstral Coefficients, and deep learning models with PyTorch. We landed on a traditional method using the Fast Fourier Transform for its relative computational efficiency and scalability, conducting a parameter space search to find an optimal configuration for determining suitable audio clips. We also sample input audio in discrete timesteps across a fixed number of buckets for increased performance and more consistent classifications.
