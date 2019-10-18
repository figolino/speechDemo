package org.tensorflow.demo;

import android.util.Log;
import android.util.Pair;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

public class RecognizeCommands {
    private static final long MINIMUM_TIME_FRACTION = 4;
    private static final String SILENCE_LABEL = "_silence_";
    private long averageWindowDurationMs;
    private float detectionThreshold;
    private List<String> labels = new ArrayList();
    private int labelsCount;
    private int minimumCount;
    private long minimumTimeBetweenSamplesMs;
    private Deque<Pair<Long, float[]>> previousResults = new ArrayDeque();
    private String previousTopLabel;
    private float previousTopLabelScore;
    private long previousTopLabelTime;
    private int suppressionMs;

    public static class RecognitionResult {
        public final String foundCommand;
        public final boolean isNewCommand;
        public final float score;

        public RecognitionResult(String inFoundCommand, float inScore, boolean inIsNewCommand) {
            this.foundCommand = inFoundCommand;
            this.score = inScore;
            this.isNewCommand = inIsNewCommand;
        }
    }

    private static class ScoreForSorting implements Comparable<ScoreForSorting> {
        public final int index;
        public final float score;

        public ScoreForSorting(float inScore, int inIndex) {
            this.score = inScore;
            this.index = inIndex;
        }

        public int compareTo(ScoreForSorting other) {
            if (this.score > other.score) {
                return -1;
            }
            if (this.score < other.score) {
                return 1;
            }
            return 0;
        }
    }

    public RecognizeCommands(List<String> inLabels, long inAverageWindowDurationMs, float inDetectionThreshold, int inSuppressionMS, int inMinimumCount, long inMinimumTimeBetweenSamplesMS) {
        this.labels = inLabels;
        this.averageWindowDurationMs = inAverageWindowDurationMs;
        this.detectionThreshold = inDetectionThreshold;
        this.suppressionMs = inSuppressionMS;
        this.minimumCount = inMinimumCount;
        this.labelsCount = inLabels.size();
        this.previousTopLabel = SILENCE_LABEL;
        this.previousTopLabelTime = Long.MIN_VALUE;
        this.previousTopLabelScore = 0.0f;
        this.minimumTimeBetweenSamplesMs = inMinimumTimeBetweenSamplesMS;
    }

    public RecognitionResult processLatestResults(float[] currentResults, long currentTimeMS) {
        if (currentResults.length != this.labelsCount) {
            throw new RuntimeException("The results for recognition should contain " + this.labelsCount + " elements, but there are " + currentResults.length);
        } else if (this.previousResults.isEmpty() || currentTimeMS >= ((Long) ((Pair) this.previousResults.getFirst()).first).longValue()) {
            int howManyResults = this.previousResults.size();
            if (howManyResults > 1 && currentTimeMS - ((Long) ((Pair) this.previousResults.getLast()).first).longValue() < this.minimumTimeBetweenSamplesMs) {
                return new RecognitionResult(this.previousTopLabel, this.previousTopLabelScore, false);
            }
            this.previousResults.addLast(new Pair(Long.valueOf(currentTimeMS), currentResults));
            long timeLimit = currentTimeMS - this.averageWindowDurationMs;
            while (((Long) ((Pair) this.previousResults.getFirst()).first).longValue() < timeLimit) {
                this.previousResults.removeFirst();
            }
            long samplesDuration = currentTimeMS - ((Long) ((Pair) this.previousResults.getFirst()).first).longValue();
            if (howManyResults < this.minimumCount || samplesDuration < this.averageWindowDurationMs / MINIMUM_TIME_FRACTION) {
                Log.v("RecognizeResult", "Too few results");
                return new RecognitionResult(this.previousTopLabel, 0.0f, false);
            }
            int i;
            boolean isNewCommand;
            float[] averageScores = new float[this.labelsCount];
            for (Pair<Long, float[]> previousResult : this.previousResults) {
                float[] scoresTensor = (float[]) previousResult.second;
                for (i = 0; i < scoresTensor.length; i++) {
                    averageScores[i] = averageScores[i] + (scoresTensor[i] / ((float) howManyResults));
                }
            }
            ScoreForSorting[] sortedAverageScores = new ScoreForSorting[this.labelsCount];
            for (i = 0; i < this.labelsCount; i++) {
                sortedAverageScores[i] = new ScoreForSorting(averageScores[i], i);
            }
            Arrays.sort(sortedAverageScores);
            String currentTopLabel = (String) this.labels.get(sortedAverageScores[0].index);
            float currentTopScore = sortedAverageScores[0].score;
            long timeSinceLastTop;
            if (this.previousTopLabel.equals(SILENCE_LABEL) || this.previousTopLabelTime == Long.MIN_VALUE) {
                timeSinceLastTop = Long.MAX_VALUE;
            } else {
                timeSinceLastTop = currentTimeMS - this.previousTopLabelTime;
            }
            if (currentTopScore <= this.detectionThreshold || timeSinceLastTop <= ((long) this.suppressionMs)) {
                isNewCommand = false;
            } else {
                this.previousTopLabel = currentTopLabel;
                this.previousTopLabelTime = currentTimeMS;
                this.previousTopLabelScore = currentTopScore;
                isNewCommand = true;
            }
            return new RecognitionResult(currentTopLabel, currentTopScore, isNewCommand);
        } else {
            throw new RuntimeException("You must feed results in increasing time order, but received a timestamp of " + currentTimeMS + " that was earlier than the previous one of " + ((Pair) this.previousResults.getFirst()).first);
        }
    }
}
