// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// ignore_for_file: public_member_api_docs

import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:convert' as convert;
import 'package:http/http.dart' as http;
void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
    // Define the default brightness and colors.
    primaryColor:const  Color(0xff110559),
     primarySwatch: MaterialColor(
  0xff110559,
  <int, Color>{
    50: Color(0xff110559),
    100: Color(0xff110559),
    200: Color(0xff110559),
    300: Color(0xff110559),
    400: Color(0xff110559),
    500: Color(0xff110559),
    600: Color(0xff110559),
    700: Color(0xff110559),
    800: Color(0xff110559),
    900: Color(0xff110559),
  }),
    // Define the default font family.
    fontFamily: 'Roboto',

    // Define the default `TextTheme`. Use this to specify the default
    // text styling for headlines, titles, bodies of text, and more.
    textTheme: const TextTheme(
      headline1: TextStyle(fontSize: 36.0, fontWeight: FontWeight.bold, color: Color(0xfff23869)),
      headline6: TextStyle(fontSize: 36.0, fontStyle: FontStyle.italic),
      bodyText2: TextStyle(fontSize: 18.0, color: Color(0xfff23869)),
    ),
  ),
      title: 'The Describer',
      home: const MyHomePage(title: 'The Describer'),
          debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, this.title}) : super(key: key);

  final String? title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  XFile? _imageFile;

  dynamic _pickImageError;
  String? _retrieveDataError;

  final ImagePicker _picker = ImagePicker();

  Future<void> _onImageButtonPressed(ImageSource source,
      {BuildContext? context}) async {
    final XFile? pickedFile = await _picker.pickImage(
      source: source,
      maxWidth: (MediaQuery.of(context!).size.width - 50),
      maxHeight: (MediaQuery.of(context).size.height - 100),
    );

    setState(() {
      _imageFile = pickedFile;
    });
     await getAIprediction();
  }

  Widget _previewImages() {
    final Text? retrieveError = _getRetrieveErrorWidget();
    if (retrieveError != null) {
      return retrieveError;
    }
    if (_imageFile != null) {
      return Semantics(
        label: 'image_picker_example_picked_image',
        child: kIsWeb
            ? Image.network(_imageFile!.path)
            : Image.file(File(_imageFile!.path)),
      );
    } else if (_pickImageError != null) {
      return Text(
        "Erreur dans le choix de l'image: $_pickImageError",
        textAlign: TextAlign.center,
      );
    } else {
      return const Text(
        "Vous n'avez pas encore choisi d'image.",
        textAlign: TextAlign.center,
      );
    }
  }

  Widget _handlePreview() {
    return _previewImages();
  }

  Future<void> retrieveLostData() async {
    final LostDataResponse response = await _picker.retrieveLostData();
    if (response.isEmpty) {
      return;
    }
    if (response.file != null) {
      if (response.type == RetrieveType.video) {
      } else {
        setState(() {
          if (response.files == null) {
            _imageFile = (response.file);
          }
        });
      }
    } else {
      _retrieveDataError = response.exception!.code;
    }
  }

  Future<String> getAIprediction() async {

    var request = http.MultipartRequest(
        'POST', Uri.parse('http://192.168.1.25:5000/iadecode/from_file'));
    request.files
        .add(http.MultipartFile.fromString('file', _imageFile!.path));

    http.StreamedResponse response = await request.send();

    if (response.statusCode == 200) {
      var message = await response.stream.bytesToString();
  
      var jsonResponse = convert.jsonDecode(message) as Map<String, dynamic>;

      var result = jsonResponse['message'];
      return result;
    } else {
      throw new Exception('Error getting response');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title!, style:Theme.of(context).textTheme.headline1 ,),
      ),
      body: Center(
        child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              _imageFile != null
                  ? FutureBuilder<String?>(
                      future: getAIprediction(),
                      builder: (BuildContext context,
                          AsyncSnapshot<String?> snapshot) {
                        switch (snapshot.connectionState) {
                          case ConnectionState.none:
                          case ConnectionState.waiting:
                            return const CircularProgressIndicator();
                          case ConnectionState.done:
                            if (!snapshot.hasData) {
                              return const Text(
                                "Erreur lors de la prédiction de l'IA2",
                                textAlign: TextAlign.center,
                              );
                            } else {
                              return Text(
                                snapshot.data!,
                                style : TextStyle(fontSize: 36.0, fontStyle: FontStyle.italic),
                                textAlign: TextAlign.center,
                              );
                            }
                          default:
                            if (snapshot.hasError) {
                              return const Text(
                                "Erreur lors de la prédiction de l'IA",
                                textAlign: TextAlign.center,
                              );
                            } else {
                              return const Text(
                                'processing',
                                textAlign: TextAlign.center,
                              );
                            }
                        }
                      })
                  : Center(),
              SizedBox(height: 20),
              !kIsWeb && defaultTargetPlatform == TargetPlatform.android
                  ? FutureBuilder<void>(
                      future: retrieveLostData(),
                      builder:
                          (BuildContext context, AsyncSnapshot<void> snapshot) {
                        switch (snapshot.connectionState) {
                          case ConnectionState.none:
                          case ConnectionState.waiting:
                            return const Text(
                              "Vous n'avez pas encore choisi d'image.",
                              textAlign: TextAlign.center,
                            );
                          case ConnectionState.done:
                            return _handlePreview();
                          default:
                            if (snapshot.hasError) {
                              return Text(
                                'Pick image/video error: ${snapshot.error}}',
                                textAlign: TextAlign.center,
                              );
                            } else {
                              return const Text(
                                "Vous n'avez pas encore choisi d'image.",
                                textAlign: TextAlign.center,
                              );
                            }
                        }
                      },
                    )
                  : _handlePreview(),
            ]),
      ),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: <Widget>[
          Semantics(
            label: 'image_picker_example_from_gallery',
            child: FloatingActionButton(
              onPressed: () {
                _onImageButtonPressed(ImageSource.gallery, context: context);
              },
              heroTag: 'image0',
              tooltip: "Choisissez une image dans la galerie",
              child: const Icon(Icons.photo),
            ),
          ),
          Padding(
            padding: const EdgeInsets.only(top: 16.0),
            child: FloatingActionButton(
              onPressed: () {
                _onImageButtonPressed(ImageSource.camera, context: context);
              },
              heroTag: 'image2',
              tooltip: "Prenez une photo",
              child: const Icon(Icons.camera_alt),
            ),
          ),
        ],
      ),
    );
  }

  Text? _getRetrieveErrorWidget() {
    if (_retrieveDataError != null) {
      final Text result = Text(_retrieveDataError!);
      _retrieveDataError = null;
      return result;
    }
    return null;
  }
}

typedef OnPickImageCallback = void Function(
    double? maxWidth, double? maxHeight, int? quality);
