# specan
This module performs acoustic analysis on a WAV file. It was inspired by the `https://github.com/primaryobjects/voice-gender/blob/master/sound.R`. It is meant to extract the same acoustic features as the sound.R module, for example for the purpose of predicting the voice gender of a new input WAV file using the ML model trained on the voice.csv dataset from `https://www.kaggle.com/primaryobjects/voicegender`.


### example ###
```
>>> from specan import specan
>>> acoustics = specan(file, duration=10)
>>> print(acoustics)

```
## Requirements

- numpy
- scipy

## License

MIT, see: LICENSE.txt

## Author

Szabolcs Szokoly <a href="mailto:sszokoly@pm.me">sszokoly@pm.me</a>
