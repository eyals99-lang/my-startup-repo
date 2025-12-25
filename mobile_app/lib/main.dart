import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Deeptech Edge',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF121212),
        colorScheme: const ColorScheme.dark(
          primary: Colors.tealAccent,
          secondary: Colors.deepPurpleAccent,
        ),
      ),
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  String _status = "System Ready";
  bool _isRecording = false;
  double _confidence = 0.0;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    // בקשת גישה למיקרופון בעלייה
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      setState(() => _status = "Mic Permission Denied");
    }
  }

  void _toggleRecording() {
    setState(() {
      _isRecording = !_isRecording;
      _status = _isRecording ? "Listening..." : "Processing Stopped";
      // בעתיד: כאן נפעיל את ה-Audio Stream
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("AI Sentinel"),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // 1. מעגל חיווי (אנימציה בעתיד)
          Center(
            child: Container(
              width: 200,
              height: 200,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _isRecording 
                    ? Colors.redAccent.withOpacity(0.1) 
                    : Colors.tealAccent.withOpacity(0.1),
                border: Border.all(
                  color: _isRecording ? Colors.redAccent : Colors.tealAccent,
                  width: 2,
                ),
                boxShadow: [
                  if (_isRecording)
                    BoxShadow(
                      color: Colors.redAccent.withOpacity(0.4),
                      blurRadius: 20,
                      spreadRadius: 5,
                    )
                ],
              ),
              child: Icon(
                _isRecording ? Icons.mic : Icons.mic_none,
                size: 80,
                color: Colors.white,
              ),
            ),
          ),
          const SizedBox(height: 50),

          // 2. תוצאות בזמן אמת
          const Text(
            "THREAT PROBABILITY",
            style: TextStyle(color: Colors.grey, letterSpacing: 2),
          ),
          const SizedBox(height: 10),
          Text(
            "${(_confidence * 100).toStringAsFixed(1)}%",
            style: const TextStyle(
              fontSize: 60, 
              fontWeight: FontWeight.bold,
              color: Colors.white
            ),
          ),
          
          const SizedBox(height: 20),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.white10,
              borderRadius: BorderRadius.circular(20)
            ),
            child: Text(
              _status.toUpperCase(),
              style: TextStyle(
                color: _isRecording ? Colors.redAccent : Colors.tealAccent,
                fontWeight: FontWeight.bold
              ),
            ),
          ),
        ],
      ),
      
      // 3. כפתור הפעלה
      floatingActionButton: FloatingActionButton.large(
        onPressed: _toggleRecording,
        backgroundColor: _isRecording ? Colors.red : Colors.teal,
        child: Icon(_isRecording ? Icons.stop : Icons.play_arrow),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}
