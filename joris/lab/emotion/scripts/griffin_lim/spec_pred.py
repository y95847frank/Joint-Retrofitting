import numpy as np
import matplotlib
matplotlib.use('Agg')
import argparse
from pylab import *
xticks([])
yticks([])
import os

import audio_utilities

# Author: Brian K. Vogel
# brian.vogel@gmail.com

model= 'bigvec_model'

def run_demo():
    """Test Griffin & Lim method for reconstructing audio from a magnitude spectrogram.

        Example of using the Griffin-Lim algorithm. The input file is loaded, the
        spectrogram is computed (note that we discard the phase information). Then,
        using only the (magnitude) spectrogram, the Griffin-Lim algorithm is run
        to reconstruct an audio signal from the spectrogram. The reconstructed audio
        is fin ally saved to a file.

        A plot of the spectrogram is also displayed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default="scripts/griffin_lim/0_short.wav",
                        help='Input WAV file')
    parser.add_argument('--sample_rate_hz', default=16000, type=int,
                        help='Sample rate in Hz')
    parser.add_argument('--fft_size', default=2048, type=int,
                        help='FFT siz')
    parser.add_argument('--iterations', default=1000, type=int,
                        help='Number of iterations to run')
    parser.add_argument('--enable_filter', action='store_true',
                        help='Apply a low-pass filter')
    parser.add_argument('--enable_mel_scale', action='store_true',
                        help='Convert to mel scale and back')
    parser.add_argument('--cutoff_freq', type=int, default=1000,
                        help='If filter is enable, the low-pass cutoff frequency in Hz')
    args = parser.parse_args()

    in_file = args.in_file

    # Load an audio file. It must be WAV format. Multi-channel files will be
    # converted to mono.
    input_signal = audio_utilities.get_signal(in_file, expected_fs=args.sample_rate_hz)

    # Hopsamp is the number of samples that the analysis window is shifted after
    # computing the FFT. For example, if the sample rate is 44100 Hz and hopsamp is
    # 256, then there will be approximately 44100/256 = 172 FFTs computed per second
    # and thus 172 spectral slices (i.e., columns) per second in the spectrogram.
    hopsamp = args.fft_size // 8

    # Compute the Short-Time Fourier Transform (STFT) from the audio file. This is a 2-dim Numpy array with
    # time_slices rows and frequency_bins columns. Thus, you will need to take the
    # transpose of this matrix to get the usual STFT which has frequency bins as rows
    # and time slices as columns.
    stft_full = audio_utilities.stft_for_reconstruction(input_signal,
                                                        args.fft_size, hopsamp)
    # Note that the STFT is complex-valued. Therefore, to get the (magnitude)
    # spectrogram, we need to take the absolute value.
    stft_mag = abs(stft_full) ** 2.0
    # Note that `stft_mag` only contains the magnitudes and so we have lost the
    # phase information.
    
    stft_mag = np.log(stft_mag + 1)
    scale = 1 / np.amax(stft_mag)
    stft_mag *= scale
    
    # We now have a (magnitude only) spectrogram, `stft_mag` that is normalized to be within [0, 1.0].
    # In a practical use case, we would probably want to perform some processing on `stft_mag` here
    # which would produce a modified version that we would want to reconstruct audio from.

    #stft_mag = np.load('/tmp/data/train_log.npy')[56]

    clf()
    figure(1)
    imshow(stft_mag.T**0.125, origin='lower', cmap=cm.hot, aspect='auto',
           interpolation='nearest', vmin=0, vmax=1)
    colorbar()
    title('Unmodified spectrum')
    xlabel('time index')
    ylabel('frequency bin index')
    savefig('unmodified_spectrogram.png', dpi=150)
    
    # load model & predict
    from keras.models import model_from_json
    latent_dim = 2048*4
    epsilon_std= 0.0
    json_file = open('checkpoints/' + model + '.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json, {'latent_dim': latent_dim, 'epsilon_std': epsilon_std})
    loaded_model.load_weights('checkpoints/' + model + '.h5')
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam')
    stft_mag = stft_mag[:, :1024]
    stft_mag = np.reshape(stft_mag, (1, 192, 1024, 1)) 
    stft_modified = loaded_model.predict(stft_mag, batch_size=1)
    stft_modified = np.reshape(stft_modified, (192, 1024))

    # If the mel scale option is selected, apply a perceptual frequency scale.
    if args.enable_mel_scale:
        min_freq_hz = 70
        max_freq_hz = 8000
        mel_bin_count = 200

        linear_bin_count = 1 + args.fft_size//2
        filterbank = audio_utilities.make_mel_filterbank(min_freq_hz, max_freq_hz, mel_bin_count,
                                                         linear_bin_count , args.sample_rate_hz)
        figure(2)
        imshow(filterbank, origin='lower', cmap=cm.hot, aspect='auto',
               interpolation='nearest')
        colorbar()
        title('Mel scale filter bank')
        xlabel('linear frequency index')
        ylabel('mel frequency index')
        savefig('mel_scale_filterbank.png', dpi=150)

        mel_spectrogram = np.dot(filterbank, stft_mag.T)

        clf()
        figure(3)
        imshow(mel_spectrogram**0.125, origin='lower', cmap=cm.hot, aspect='auto',
               interpolation='nearest')
        colorbar()
        title('Mel scale spectrogram')
        xlabel('time index')
        ylabel('mel frequency bin index')
        savefig('mel_scale_spectrogram.png', dpi=150)

        inverted_mel_to_linear_freq_spectrogram = np.dot(filterbank.T, mel_spectrogram)

        clf()
        figure(4)
        imshow(inverted_mel_to_linear_freq_spectrogram**0.125, origin='lower', cmap=cm.hot, aspect='auto',
               interpolation='nearest')
        colorbar()
        title('Linear scale spectrogram obtained from mel scale spectrogram')
        xlabel('time index')
        ylabel('frequency bin index')
        savefig('inverted_mel_to_linear_freq_spectrogram.png', dpi=150)

        stft_modified = inverted_mel_to_linear_freq_spectrogram.T

    ###### Optional: modify the spectrogram
    # For example, we can implement a low-pass filter by simply setting all frequency bins above
    # some threshold frequency (args.cutoff_freq) to 0 as follows.
    if args.enable_filter:
        # Calculate corresponding bin index.
        cutoff_bin = round(args.cutoff_freq*args.fft_size/args.sample_rate_hz)
        stft_modified[:, cutoff_bin:] = 0
    ###########
    
    clf()
    figure(5)
    imshow(stft_modified.T**0.125, origin='lower', cmap=cm.hot, aspect='auto',
           interpolation='nearest', vmin=0, vmax=1)
    colorbar()
    title('Spectrogram used to reconstruct audio')
    xlabel('time index')
    ylabel('frequency bin index')
    savefig('reconstruction_spectrogram.png', dpi=150)

    # Undo the rescaling.
    stft_modified = np.e ** stft_modified - 1
    stft_modified_scaled = stft_modified
    
    # Use the Griffin&Lim algorithm to reconstruct an audio signal from the
    # magnitude spectrogram.
    add_col = stft_modified_scaled[:,-1]
    add_col = np.reshape(add_col, (192, 1))
    print(add_col.shape)
    stft_modified_scaled = np.concatenate((stft_modified_scaled, add_col), axis=1)
    print(stft_modified_scaled.shape)
    x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(stft_modified_scaled,
                                                                   args.fft_size, hopsamp,
                                                                   args.iterations)
    # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    max_sample = np.max(abs(x_reconstruct))
    if max_sample > 1.0:
        x_reconstruct = x_reconstruct / max_sample

    # Save the reconstructed signal to a WAV file.
    audio_utilities.save_audio_to_file(x_reconstruct, args.sample_rate_hz)
 
    # Save the spectrogram image also.
   
if __name__ == '__main__':
    run_demo()
