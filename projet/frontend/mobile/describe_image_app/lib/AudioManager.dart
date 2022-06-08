import 'package:flutter_tts/flutter_tts.dart';
import 'dart:io' show Platform;
class AudioManager {
  AudioManager._() {
 
    FlutterTts().setLanguage(Platform.localeName);
  }

  static final AudioManager _instance = AudioManager._();

  factory AudioManager() {
    return _instance;
  }

  speak(String text) {
    FlutterTts().speak(text);
  }
}


