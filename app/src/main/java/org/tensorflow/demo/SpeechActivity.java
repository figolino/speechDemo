package org.tensorflow.demo;

import android.animation.AnimatorInflater;
import android.animation.AnimatorSet;
import android.app.Activity;
import android.media.AudioRecord;
import android.os.Build.VERSION;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.RecognizeCommands.RecognitionResult;

import ch.lovisi.zhaw.speech.R;

public class SpeechActivity extends Activity {
    private static final long AVERAGE_WINDOW_DURATION_MS = 500;
    private static final float DETECTION_THRESHOLD = 0.7f;
    private static final String INPUT_DATA_NAME = "decoded_sample_data:0";
    private static final String LABEL_FILENAME = "file:///android_asset/conv_actions_labels.txt";
    private static final String LOG_TAG = SpeechActivity.class.getSimpleName();
    private static final int MINIMUM_COUNT = 3;
    private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30;
    private static final String MODEL_FILENAME = "file:///android_asset/conv_actions_frozen.pb";
    private static final String OUTPUT_SCORES_NAME = "labels_softmax";
    private static final int RECORDING_LENGTH = 16000;
    private static final int REQUEST_RECORD_AUDIO = 13;
    private static final int SAMPLE_DURATION_MS = 1000;
    private static final int SAMPLE_RATE = 16000;
    private static final String SAMPLE_RATE_NAME = "decoded_sample_data:1";
    private static final int SUPPRESSION_MS = 1500;
    private List<String> displayedLabels = new ArrayList();
    private TensorFlowInferenceInterface inferenceInterface;
    private List<String> labels = new ArrayList();
    private ListView labelsListView;
    private Button quitButton;
    private Thread recognitionThread;
    private RecognizeCommands recognizeCommands = null;
    short[] recordingBuffer = new short[16000];
    private final ReentrantLock recordingBufferLock = new ReentrantLock();
    int recordingOffset = 0;
    private Thread recordingThread;
    boolean shouldContinue = true;
    boolean shouldContinueRecognition = true;

    protected void onCreate(Bundle savedInstanceState) {
        IOException e;
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_speech);
        this.quitButton = (Button) findViewById(R.id.quit);
        this.quitButton.setOnClickListener(new OnClickListener() {
            public void onClick(View view) {
                SpeechActivity.this.moveTaskToBack(true);
                Process.killProcess(Process.myPid());
                System.exit(1);
            }
        });
        this.labelsListView = (ListView) findViewById(R.id.list_view);
        String actualFilename = LABEL_FILENAME.split("file:///android_asset/")[1];
        Log.i(LOG_TAG, "Reading labels from: " + actualFilename);
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            while (true) {
                try {
                    String line = br.readLine();
                    if (line != null) {
                        this.labels.add(line);
                        if (line.charAt(0) != '_') {
                            this.displayedLabels.add(line.substring(0, 1).toUpperCase() + line.substring(1));
                        }
                    } else {
                        br.close();
                        this.labelsListView.setAdapter(new ArrayAdapter(this, R.layout.list_text_item, this.displayedLabels));
                        this.recognizeCommands = new RecognizeCommands(this.labels, AVERAGE_WINDOW_DURATION_MS, DETECTION_THRESHOLD, SUPPRESSION_MS, MINIMUM_COUNT, MINIMUM_TIME_BETWEEN_SAMPLES_MS);
                        this.inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILENAME);
                        requestMicrophonePermission();
                        startRecording();
                        startRecognition();
                        return;
                    }
                } catch (IOException e2) {
                    e = e2;
                    BufferedReader bufferedReader = br;
                }
            }
        } catch (IOException e3) {
            e = e3;
            throw new RuntimeException("Problem reading label file!", e);
        }
    }

    private void requestMicrophonePermission() {
        if (VERSION.SDK_INT >= 23) {
            requestPermissions(new String[]{"android.permission.RECORD_AUDIO"}, REQUEST_RECORD_AUDIO);
        }
    }

    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.length > 0 && grantResults[0] == 0) {
            startRecording();
            startRecognition();
        }
    }

    public synchronized void startRecording() {
        if (this.recordingThread == null) {
            this.shouldContinue = true;
            this.recordingThread = new Thread(new Runnable() {
                public void run() {
                    SpeechActivity.this.record();
                }
            });
            this.recordingThread.start();
        }
    }

    public synchronized void stopRecording() {
        if (this.recordingThread != null) {
            this.shouldContinue = false;
            this.recordingThread = null;
        }
    }

    private void record() {
        Process.setThreadPriority(-16);
        int bufferSize = AudioRecord.getMinBufferSize(16000, 16, 2);
        if (bufferSize == -1 || bufferSize == -2) {
            bufferSize = 32000;
        }
        short[] audioBuffer = new short[(bufferSize / 2)];
        AudioRecord record = new AudioRecord(0, 16000, 16, 2, bufferSize);
        if (record.getState() != 1) {
            Log.e(LOG_TAG, "Audio Record can't initialize!");
            return;
        }
        record.startRecording();
        Log.v(LOG_TAG, "Start recording");
        while (this.shouldContinue) {
            int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
            int maxLength = this.recordingBuffer.length;
            int newRecordingOffset = this.recordingOffset + numberRead;
            int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
            int firstCopyLength = numberRead - secondCopyLength;
            this.recordingBufferLock.lock();
            try {
                System.arraycopy(audioBuffer, 0, this.recordingBuffer, this.recordingOffset, firstCopyLength);
                System.arraycopy(audioBuffer, firstCopyLength, this.recordingBuffer, 0, secondCopyLength);
                this.recordingOffset = newRecordingOffset % maxLength;
                this.recordingBufferLock.unlock();
            } catch (Throwable th) {
                this.recordingBufferLock.unlock();
            }
        }
        record.stop();
        record.release();
    }

    public synchronized void startRecognition() {
        if (this.recognitionThread == null) {
            this.shouldContinueRecognition = true;
            this.recognitionThread = new Thread(new Runnable() {
                public void run() {
                    SpeechActivity.this.recognize();
                }
            });
            this.recognitionThread.start();
        }
    }

    public synchronized void stopRecognition() {
        if (this.recognitionThread != null) {
            this.shouldContinueRecognition = false;
            this.recognitionThread = null;
        }
    }

    private void recognize() {
        Log.v(LOG_TAG, "Start recognition");
        short[] inputBuffer = new short[16000];
        float[] floatInputBuffer = new float[16000];
        float[] outputScores = new float[this.labels.size()];
        String[] outputScoresNames = new String[]{OUTPUT_SCORES_NAME};
        int[] sampleRateList = new int[]{16000};
        while (this.shouldContinueRecognition) {
            this.recordingBufferLock.lock();
            try {
                int firstCopyLength = this.recordingBuffer.length - this.recordingOffset;
                int secondCopyLength = this.recordingOffset;
                System.arraycopy(this.recordingBuffer, this.recordingOffset, inputBuffer, 0, firstCopyLength);
                System.arraycopy(this.recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
                for (int i = 0; i < 16000; i++) {
                    floatInputBuffer[i] = ((float) inputBuffer[i]) / 32767.0f;
                }
                this.inferenceInterface.feed(SAMPLE_RATE_NAME, sampleRateList, new long[0]);
                this.inferenceInterface.feed(INPUT_DATA_NAME, floatInputBuffer, 16000, 1);
                this.inferenceInterface.run(outputScoresNames);
                this.inferenceInterface.fetch(OUTPUT_SCORES_NAME, outputScores);
                final RecognitionResult result = this.recognizeCommands.processLatestResults(outputScores, System.currentTimeMillis());
                runOnUiThread(new Runnable() {
                    public void run() {
                        if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
                            int labelIndex = -1;
                            for (int i = 0; i < SpeechActivity.this.labels.size(); i++) {
                                if (((String) SpeechActivity.this.labels.get(i)).equals(result.foundCommand)) {
                                    labelIndex = i;
                                }
                            }
                            AnimatorSet colorAnimation = (AnimatorSet) AnimatorInflater.loadAnimator(SpeechActivity.this, R.animator.color_animation);
                            colorAnimation.setTarget(SpeechActivity.this.labelsListView.getChildAt(labelIndex - 2));
                            colorAnimation.start();
                            Toast.makeText(getApplicationContext(),"score: "+result.score,Toast.LENGTH_LONG).show();
                        }
                    }
                });
                try {
                    Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
                } catch (InterruptedException e) {
                    Log.e(LOG_TAG, Objects.requireNonNull(e.getMessage()));
                }
            } finally {
                this.recordingBufferLock.unlock();
            }
        }
        Log.v(LOG_TAG, "End recognition");
    }
}
