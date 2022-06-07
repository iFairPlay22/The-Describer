import 'package:flutter_tts/flutter_tts.dart';

class AudioManager {
  AudioManager._() {
    FlutterTts().setLanguage("en-us");
  }

  static final AudioManager _instance = AudioManager._();

  factory AudioManager() {
    return _instance;
  }

  speak(String text) {
    FlutterTts().speak(text);
  }
}


