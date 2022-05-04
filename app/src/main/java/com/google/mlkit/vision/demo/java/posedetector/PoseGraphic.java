/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java.posedetector;

import static java.lang.Math.max;
import static java.lang.Math.min;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.google.common.primitives.Ints;
import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;

import java.util.List;
import java.util.Locale;

/**
 * Draw the detected pose in preview.
 */
public class PoseGraphic extends Graphic {

    /**
     * my own code
     **/
    static double getAngle(PoseLandmark firstPoint, PoseLandmark midPoint, PoseLandmark lastPoint) {
        double result =
                Math.toDegrees(
                        java.lang.Math.atan2(lastPoint.getPosition().y - midPoint.getPosition().y,
                                lastPoint.getPosition().x - midPoint.getPosition().x)
                                - java.lang.Math.atan2(firstPoint.getPosition().y - midPoint.getPosition().y,
                                firstPoint.getPosition().x - midPoint.getPosition().x));
        result = Math.abs(result); // Angle should never be negative
        if (result > 180) {
            result = (360.0 - result); // Always get the acute representation of the angle
        }
        return result;
    }

    static double getAngle3D(PoseLandmark firstPoint, PoseLandmark midPoint, PoseLandmark lastPoint) {
        float Ax = firstPoint.getPosition3D().getX();
        float Ay = firstPoint.getPosition3D().getY();
        float Az = firstPoint.getPosition3D().getZ();

        float Bx = midPoint.getPosition3D().getX();
        float By = midPoint.getPosition3D().getY();
        float Bz = midPoint.getPosition3D().getZ();

        float Cx = lastPoint.getPosition3D().getX();
        float Cy = lastPoint.getPosition3D().getY();
        float Cz = lastPoint.getPosition3D().getZ();

//      float Ax = 3;
//      float Ay = 0;
//      float Az = 0;
//
//      float Bx = 3;
//      float By = 5;
//      float Bz = 0;
//
//      float Cx = 3;
//      float Cy = 10;
//      float Cz = 0;

        // Find direction ratio of line AB
        float v1x = Ax - Bx;
        float v1y = Ay - By;
        float v1z = Az - Bz;
//      System.out.println("v1x: " + v1x);
//      System.out.println("v1y: " + v1y);
//      System.out.println("v1z: " + v1z);

        // Find direction ratio of line BC
        double v2x = Cx - Bx;
        double v2y = Cy - By;
        double v2z = Cz - Bz;
//    System.out.println("v2x: " + v2x);
//    System.out.println("v2y: " + v2y);
//    System.out.println("v2z: " + v2z);

        double v1mag = Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
        double v2mag = Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
//    System.out.println("v1mag: " + v1mag);
//    System.out.println("v2mag: " + v2mag);

        double v1normx = v1x / v1mag;
        double v1normy = v1y / v1mag;
        double v1normz = v1z / v1mag;
//        System.out.println("v1normx: " + v1normx);
//        System.out.println("v1normy: " + v1normy);
//        System.out.println("v1normz: " + v1normz);

        double v2normx = v2x / v2mag;
        double v2normy = v2y / v2mag;
        double v2normz = v2z / v2mag;

        double res = v1normx * v2normx + v1normy * v2normy + v1normz * v2normz;

        // Print the angle
        return Math.acos(res) * 180 / 3.141592653589793;

    }

    private static final float DOT_RADIUS = 8.0f;
    private static final float IN_FRAME_LIKELIHOOD_TEXT_SIZE = 30.0f;
    private static final float STROKE_WIDTH = 4.0f;
    // changes bottom text "squats_up: 1.00 confidence" etc.
    private static final float POSE_CLASSIFICATION_TEXT_SIZE = 30.0f;

    private final Pose pose;
    private final boolean showAngles;
    private final boolean visualizeZ;
    private final boolean rescaleZForVisualization;
    private float zMin = Float.MAX_VALUE;
    private float zMax = Float.MIN_VALUE;

    private final List<String> poseClassification;
    private final Paint classificationTextPaint;
    private final Paint betterFormPaint;
    private final Paint leftPaint;
    private final Paint rightPaint;
    private final Paint whitePaint;

    /**
     my own code
     **/
    private final Paint blueAnglePaint;
    private final Paint greenAnglePaint;
    private final Paint redAnglePaint;
    private final Paint yellowAnglePaint;

    PoseGraphic(
            GraphicOverlay overlay,
            Pose pose,
            boolean showAngles,
            boolean visualizeZ,
            boolean rescaleZForVisualization,
            List<String> poseClassification) {
        super(overlay);
        this.pose = pose;
        this.showAngles = showAngles;
        this.visualizeZ = visualizeZ;
        this.rescaleZForVisualization = rescaleZForVisualization;
        this.poseClassification = poseClassification;

        classificationTextPaint = new Paint();
        classificationTextPaint.setColor(Color.WHITE);
        classificationTextPaint.setTextSize(POSE_CLASSIFICATION_TEXT_SIZE);
        classificationTextPaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK);

        whitePaint = new Paint();
        whitePaint.setStrokeWidth(STROKE_WIDTH);
        whitePaint.setColor(Color.WHITE);
        whitePaint.setTextSize(IN_FRAME_LIKELIHOOD_TEXT_SIZE);
        leftPaint = new Paint();
        leftPaint.setStrokeWidth(STROKE_WIDTH);
        leftPaint.setColor(Color.GREEN);
        rightPaint = new Paint();
        rightPaint.setStrokeWidth(STROKE_WIDTH);
        rightPaint.setColor(Color.YELLOW);

        /** my own code **/
        betterFormPaint = new Paint();
        betterFormPaint.setColor(Color.WHITE);
        betterFormPaint.setTextSize(45.0f);
        betterFormPaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK);

        blueAnglePaint = new Paint();
        blueAnglePaint.setColor(Color.rgb(102, 178, 255));
        blueAnglePaint.setTextSize(40.0f);
        blueAnglePaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK);

        greenAnglePaint = new Paint();
        greenAnglePaint.setColor(Color.rgb(144, 238, 144));
        greenAnglePaint.setTextSize(40.0f);
        greenAnglePaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK);

        redAnglePaint = new Paint();
        redAnglePaint.setColor(Color.rgb(222, 23, 56));
        redAnglePaint.setTextSize(40.0f);
        redAnglePaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK);

        yellowAnglePaint = new Paint();
        yellowAnglePaint.setColor(Color.rgb(255, 204, 0));
        yellowAnglePaint.setTextSize(40.0f);
        yellowAnglePaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK);
    }


    @Override
    public void draw(Canvas canvas) {
        List<PoseLandmark> landmarks = pose.getAllPoseLandmarks();
        if (landmarks.isEmpty()) {
            return;
        }

        // Draw pose classification text.
//    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
//
//    for (int i = 0; i < poseClassification.size(); i++) {
//      float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
//          * (poseClassification.size() - i));
//      canvas.drawText(
//          poseClassification.get(i),
//          classificationX,
//          classificationY,
//          classificationTextPaint);
//    }

        float x = (canvas.getWidth() - POSE_CLASSIFICATION_TEXT_SIZE * 16.0f);


        for (int i = 0; i < poseClassification.size(); i++) {
            float y = (POSE_CLASSIFICATION_TEXT_SIZE * 1.5f * (poseClassification.size() - i));
            canvas.drawText(
                    poseClassification.get(i),
                    x,
                    y,
                    classificationTextPaint);
        }


        // Draw all the points
//    for (PoseLandmark landmark : landmarks) {
//      drawPoint(canvas, landmark, whitePaint);
//      if (visualizeZ && rescaleZForVisualization) {
//        zMin = min(zMin, landmark.getPosition3D().getZ());
//        zMax = max(zMax, landmark.getPosition3D().getZ());
//      }
//    }

        /** my own code to remove points drawn on face **/
        for (int i = 11; i <= 32; i++) {
            drawPoint(canvas, landmarks.get(i), whitePaint);
            if (visualizeZ && rescaleZForVisualization) {
                zMin = min(zMin, landmarks.get(i).getPosition3D().getZ());
                zMax = max(zMax, landmarks.get(i).getPosition3D().getZ());
            }
        }

        // Face
        PoseLandmark nose = pose.getPoseLandmark(PoseLandmark.NOSE);
        PoseLandmark lefyEyeInner = pose.getPoseLandmark(PoseLandmark.LEFT_EYE_INNER);
        PoseLandmark lefyEye = pose.getPoseLandmark(PoseLandmark.LEFT_EYE);
        PoseLandmark leftEyeOuter = pose.getPoseLandmark(PoseLandmark.LEFT_EYE_OUTER);
        PoseLandmark rightEyeInner = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE_INNER);
        PoseLandmark rightEye = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE);
        PoseLandmark rightEyeOuter = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE_OUTER);
        PoseLandmark leftEar = pose.getPoseLandmark(PoseLandmark.LEFT_EAR);
        PoseLandmark rightEar = pose.getPoseLandmark(PoseLandmark.RIGHT_EAR);
        PoseLandmark leftMouth = pose.getPoseLandmark(PoseLandmark.LEFT_MOUTH);
        PoseLandmark rightMouth = pose.getPoseLandmark(PoseLandmark.RIGHT_MOUTH);

        PoseLandmark leftShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER);
        PoseLandmark rightShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER);
        PoseLandmark leftElbow = pose.getPoseLandmark(PoseLandmark.LEFT_ELBOW);
        PoseLandmark rightElbow = pose.getPoseLandmark(PoseLandmark.RIGHT_ELBOW);
        PoseLandmark leftWrist = pose.getPoseLandmark(PoseLandmark.LEFT_WRIST);
        PoseLandmark rightWrist = pose.getPoseLandmark(PoseLandmark.RIGHT_WRIST);
        PoseLandmark leftHip = pose.getPoseLandmark(PoseLandmark.LEFT_HIP);
        PoseLandmark rightHip = pose.getPoseLandmark(PoseLandmark.RIGHT_HIP);
        PoseLandmark leftKnee = pose.getPoseLandmark(PoseLandmark.LEFT_KNEE);
        PoseLandmark rightKnee = pose.getPoseLandmark(PoseLandmark.RIGHT_KNEE);
        PoseLandmark leftAnkle = pose.getPoseLandmark(PoseLandmark.LEFT_ANKLE);
        PoseLandmark rightAnkle = pose.getPoseLandmark(PoseLandmark.RIGHT_ANKLE);

        PoseLandmark leftPinky = pose.getPoseLandmark(PoseLandmark.LEFT_PINKY);
        PoseLandmark rightPinky = pose.getPoseLandmark(PoseLandmark.RIGHT_PINKY);
        PoseLandmark leftIndex = pose.getPoseLandmark(PoseLandmark.LEFT_INDEX);
        PoseLandmark rightIndex = pose.getPoseLandmark(PoseLandmark.RIGHT_INDEX);
        PoseLandmark leftThumb = pose.getPoseLandmark(PoseLandmark.LEFT_THUMB);
        PoseLandmark rightThumb = pose.getPoseLandmark(PoseLandmark.RIGHT_THUMB);
        PoseLandmark leftHeel = pose.getPoseLandmark(PoseLandmark.LEFT_HEEL);
        PoseLandmark rightHeel = pose.getPoseLandmark(PoseLandmark.RIGHT_HEEL);
        PoseLandmark leftFootIndex = pose.getPoseLandmark(PoseLandmark.LEFT_FOOT_INDEX);
        PoseLandmark rightFootIndex = pose.getPoseLandmark(PoseLandmark.RIGHT_FOOT_INDEX);

        // Face [commented out to remove lines on face]
//    drawLine(canvas, nose, lefyEyeInner, whitePaint);
//    drawLine(canvas, lefyEyeInner, lefyEye, whitePaint);
//    drawLine(canvas, lefyEye, leftEyeOuter, whitePaint);
//    drawLine(canvas, leftEyeOuter, leftEar, whitePaint);
//    drawLine(canvas, nose, rightEyeInner, whitePaint);
//    drawLine(canvas, rightEyeInner, rightEye, whitePaint);
//    drawLine(canvas, rightEye, rightEyeOuter, whitePaint);
//    drawLine(canvas, rightEyeOuter, rightEar, whitePaint);
//    drawLine(canvas, leftMouth, rightMouth, whitePaint);

        drawLine(canvas, leftShoulder, rightShoulder, whitePaint);
        drawLine(canvas, leftHip, rightHip, whitePaint);

        // Left body
        drawLine(canvas, leftShoulder, leftElbow, leftPaint);
        drawLine(canvas, leftElbow, leftWrist, leftPaint);
        drawLine(canvas, leftShoulder, leftHip, leftPaint);
        drawLine(canvas, leftHip, leftKnee, leftPaint);
        drawLine(canvas, leftKnee, leftAnkle, leftPaint);
        drawLine(canvas, leftWrist, leftThumb, leftPaint);
        drawLine(canvas, leftWrist, leftPinky, leftPaint);
        drawLine(canvas, leftWrist, leftIndex, leftPaint);
        drawLine(canvas, leftIndex, leftPinky, leftPaint);
        drawLine(canvas, leftAnkle, leftHeel, leftPaint);
        drawLine(canvas, leftHeel, leftFootIndex, leftPaint);

        // Right body
        drawLine(canvas, rightShoulder, rightElbow, rightPaint);
        drawLine(canvas, rightElbow, rightWrist, rightPaint);
        drawLine(canvas, rightShoulder, rightHip, rightPaint);
        drawLine(canvas, rightHip, rightKnee, rightPaint);
        drawLine(canvas, rightKnee, rightAnkle, rightPaint);
        drawLine(canvas, rightWrist, rightThumb, rightPaint);
        drawLine(canvas, rightWrist, rightPinky, rightPaint);
        drawLine(canvas, rightWrist, rightIndex, rightPaint);
        drawLine(canvas, rightIndex, rightPinky, rightPaint);
        drawLine(canvas, rightAnkle, rightHeel, rightPaint);
        drawLine(canvas, rightHeel, rightFootIndex, rightPaint);

        /**
         // Draw inFrameLikelihood for all points
         if (showAngles) {
         for (PoseLandmark landmark : landmarks) {
         canvas.drawText(
         String.format(Locale.US, "%.2f", landmark.getInFrameLikelihood()),
         translateX(landmark.getPosition().x),
         translateY(landmark.getPosition().y),
         whitePaint);
         }
         }
         **/


//        float leftElbowAngle = (float) getAngle(leftShoulder, leftElbow, leftWrist);
//        float rightElbowAngle = (float) getAngle(rightShoulder, rightElbow, rightWrist);
//        float leftArmpitAngle = (float) getAngle(leftElbow, leftShoulder, leftHip);
//        float rightArmpitAngle = (float) getAngle(rightElbow, rightShoulder, rightHip);
//      float leftHipAngle = (float) getAngle( leftShoulder, leftHip, leftKnee );
//      float rightHipAngle = (float) getAngle( rightShoulder, rightHip, rightKnee );

//      tester
        float leftElbowAngle = (float) getAngle(leftShoulder, leftElbow, leftWrist);
        float rightElbowAngle = (float) getAngle(rightShoulder, rightElbow, rightWrist);
        float leftArmpitAngle = (float) getAngle(leftElbow, leftShoulder, leftHip);
        float rightArmpitAngle = (float) getAngle(rightElbow, rightShoulder, rightHip);
        float leftHipAngle = (float) getAngle(leftShoulder, leftHip, leftKnee);
        float rightHipAngle = (float) getAngle(rightShoulder, rightHip, rightKnee);
        float leftWristAngle = (float) getAngle(leftElbow, leftWrist, leftThumb);
        float rightWristAngle = (float) getAngle(rightElbow, rightWrist, rightThumb);
        float leftNeckAngle = (float) getAngle(leftHip, leftShoulder, leftMouth);
        float rightNeckAngle = (float) getAngle(rightHip, rightShoulder, rightMouth);
        float leftKneeAngle = (float) getAngle(leftHip, leftKnee, leftAnkle);
        float rightKneeAngle = (float) getAngle(rightHip, rightKnee, rightAnkle);

        /** my own code **/
        if (showAngles) {
            if (!poseClassification.isEmpty()) {
                if (poseClassification.get(0).contains("bicepcurls")) {
                    // left elbow angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftElbowAngle),
                            translateX(leftElbow.getPosition().x),
                            translateY(leftElbow.getPosition().y),
                            blueAnglePaint);

                    // right elbow angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightElbowAngle),
                            translateX(rightElbow.getPosition().x),
                            translateY(rightElbow.getPosition().y),
                            blueAnglePaint);

                    // left armpit angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftArmpitAngle),
                            translateX(leftShoulder.getPosition().x),
                            translateY(leftShoulder.getPosition().y),
                            blueAnglePaint);

                    // right armpit angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightArmpitAngle),
                            translateX(rightShoulder.getPosition().x),
                            translateY(rightShoulder.getPosition().y),
                            blueAnglePaint);

                    // left hip angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftHipAngle),
                            translateX(leftHip.getPosition().x),
                            translateY(leftHip.getPosition().y),
                            blueAnglePaint);

                    // right hip angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightHipAngle),
                            translateX(rightHip.getPosition().x),
                            translateY(rightHip.getPosition().y),
                            blueAnglePaint);

//                algorithm does a really poor attempt at detecting hand and thumb so can't detect form for wrist
//                    // left wrist angle
//                    canvas.drawText(
//                            String.format(Locale.US, "%.0f", leftWristAngle),
//                            translateX(leftWrist.getPosition().x),
//                            translateY(leftWrist.getPosition().y),
//                            blueAnglePaint);
//
//                    // right wrist angle
//                    canvas.drawText(
//                            String.format(Locale.US, "%.0f", rightWristAngle),
//                            translateX(rightWrist.getPosition().x),
//                            translateY(rightWrist.getPosition().y),
//                            blueAnglePaint);
                }

                if (poseClassification.get(0).contains("pushups")) {
                    // left elbow angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftElbowAngle),
                            translateX(leftElbow.getPosition().x),
                            translateY(leftElbow.getPosition().y),
                            greenAnglePaint);

                    // right elbow angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightElbowAngle),
                            translateX(rightElbow.getPosition().x),
                            translateY(rightElbow.getPosition().y),
                            greenAnglePaint);

                    // left armpit angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftArmpitAngle),
                            translateX(leftShoulder.getPosition().x),
                            translateY(leftShoulder.getPosition().y),
                            greenAnglePaint);

                    // right armpit angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightArmpitAngle),
                            translateX(rightShoulder.getPosition().x),
                            translateY(rightShoulder.getPosition().y),
                            greenAnglePaint);

                    // left hip angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftHipAngle),
                            translateX(leftHip.getPosition().x),
                            translateY(leftHip.getPosition().y),
                            greenAnglePaint);

                    // right hip angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightHipAngle),
                            translateX(rightHip.getPosition().x),
                            translateY(rightHip.getPosition().y),
                            greenAnglePaint);

                    // left neck angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftNeckAngle),
                            translateX(leftMouth.getPosition().x),
                            translateY(leftMouth.getPosition().y),
                            greenAnglePaint);

                    // right neck angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightNeckAngle),
                            translateX(rightMouth.getPosition().x),
                            translateY(rightMouth.getPosition().y),
                            greenAnglePaint);
                }

                if (poseClassification.get(0).contains("squats")) {
                    // left neck angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftNeckAngle),
                            translateX(leftMouth.getPosition().x),
                            translateY(leftMouth.getPosition().y),
                            redAnglePaint);

                    // right neck angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightNeckAngle),
                            translateX(rightMouth.getPosition().x),
                            translateY(rightMouth.getPosition().y),
                            redAnglePaint);

                    // left armpit angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftArmpitAngle),
                            translateX(leftShoulder.getPosition().x),
                            translateY(leftShoulder.getPosition().y),
                            redAnglePaint);

                    // right armpit angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightArmpitAngle),
                            translateX(rightShoulder.getPosition().x),
                            translateY(rightShoulder.getPosition().y),
                            redAnglePaint);

                    // left elbow angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftElbowAngle),
                            translateX(leftElbow.getPosition().x),
                            translateY(leftElbow.getPosition().y),
                            redAnglePaint);

                    // right elbow angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightElbowAngle),
                            translateX(rightElbow.getPosition().x),
                            translateY(rightElbow.getPosition().y),
                            redAnglePaint);

                    // left hip angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftHipAngle),
                            translateX(leftHip.getPosition().x),
                            translateY(leftHip.getPosition().y),
                            redAnglePaint);

                    // right hip angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightHipAngle),
                            translateX(rightHip.getPosition().x),
                            translateY(rightHip.getPosition().y),
                            redAnglePaint);

                    // left knee angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftKneeAngle),
                            translateX(leftKnee.getPosition().x),
                            translateY(leftKnee.getPosition().y),
                            redAnglePaint);

                    // right knee angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightKneeAngle),
                            translateX(rightKnee.getPosition().x),
                            translateY(rightKnee.getPosition().y),
                            redAnglePaint);
                }

                if (poseClassification.get(0).contains("sumo")) {
                    // left neck angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftNeckAngle),
                            translateX(leftMouth.getPosition().x),
                            translateY(leftMouth.getPosition().y),
                            yellowAnglePaint);

                    // right neck angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightNeckAngle),
                            translateX(rightMouth.getPosition().x),
                            translateY(rightMouth.getPosition().y),
                            yellowAnglePaint);

                    // left armpit angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftArmpitAngle),
                            translateX(leftShoulder.getPosition().x),
                            translateY(leftShoulder.getPosition().y),
                            yellowAnglePaint);

                    // right armpit angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightArmpitAngle),
                            translateX(rightShoulder.getPosition().x),
                            translateY(rightShoulder.getPosition().y),
                            yellowAnglePaint);

                    // left elbow angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftElbowAngle),
                            translateX(leftElbow.getPosition().x),
                            translateY(leftElbow.getPosition().y),
                            yellowAnglePaint);

                    // right elbow angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightElbowAngle),
                            translateX(rightElbow.getPosition().x),
                            translateY(rightElbow.getPosition().y),
                            yellowAnglePaint);

                    // left hip angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftHipAngle),
                            translateX(leftHip.getPosition().x),
                            translateY(leftHip.getPosition().y),
                            yellowAnglePaint);

                    // right hip angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightHipAngle),
                            translateX(rightHip.getPosition().x),
                            translateY(rightHip.getPosition().y),
                            yellowAnglePaint);

                    // left knee angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", leftKneeAngle),
                            translateX(leftKnee.getPosition().x),
                            translateY(leftKnee.getPosition().y),
                            yellowAnglePaint);

                    // right knee angle
                    canvas.drawText(
                            String.format(Locale.US, "%.0f", rightKneeAngle),
                            translateX(rightKnee.getPosition().x),
                            translateY(rightKnee.getPosition().y),
                            yellowAnglePaint);
                }

            }
        }


        if (!poseClassification.isEmpty()) {
            if (poseClassification.get(0).contains("bicepcurls")) {

                if (leftArmpitAngle > 40f || rightArmpitAngle > 40f) {
                    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                    float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                            * 3);
                    canvas.drawText(
                            "Keep your elbows close to your body",
                            classificationX,
                            classificationY,
                            betterFormPaint);
                }

                if (leftHipAngle < 160f || rightHipAngle < 160f) {
                    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                    float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                            * 4);
                    canvas.drawText(
                            "Don't swing your body",
                            classificationX,
                            classificationY,
                            betterFormPaint);
                }

//                algorithm does a really poor attempt at detecting hand and thumb so can't detect form for wrist
//                if (leftWristAngle < 110f || rightWristAngle < 110f) {
//                    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
//                    float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
//                            * 1);
//                    canvas.drawText(
//                            "Don't bend your wrist",
//                            classificationX,
//                            classificationY,
//                            betterFormPaint);
//                }
            }

            if (poseClassification.get(0).contains("pushup")) {

                if (poseClassification.get(0).contains("pushups_down")) {
                    if (leftArmpitAngle > 70f || rightArmpitAngle > 70f) {
                        float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                        float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                                * 3);
                        canvas.drawText(
                                "Don't flare out your elbows",
                                classificationX,
                                classificationY,
                                betterFormPaint);
                    }
                }

                if (leftHipAngle < 165f || rightHipAngle < 165f) {
                    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                    float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                            * 4);
                    canvas.drawText(
                            "Keep your body and legs align",
                            classificationX,
                            classificationY,
                            betterFormPaint);
                }

                if (leftNeckAngle < 120f || rightNeckAngle < 120f) {
                    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                    float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                            * 5);
                    canvas.drawText(
                            "Keep your head straight",
                            classificationX,
                            classificationY,
                            betterFormPaint);
                }


//                algorithm does a really poor attempt at detecting hand and thumb so can't detect form for wrist
//                if (leftWristAngle < 110f || rightWristAngle < 110f) {
//                    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
//                    float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
//                            * 1);
//                    canvas.drawText(
//                            "Don't bend your wrist",
//                            classificationX,
//                            classificationY,
//                            betterFormPaint);
//                }

            }

            if (poseClassification.get(0).contains("squats")) {
                if (poseClassification.get(0).contains("squats_up")) {
                    if (leftArmpitAngle > 85f || rightArmpitAngle > 85f) {
                        float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                        float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                                * 3);
                        canvas.drawText(
                                "Keep your elbows close to your body",
                                classificationX,
                                classificationY,
                                betterFormPaint);
                    }

                    if (leftElbowAngle > 85f || rightElbowAngle > 85f) {
                        float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                        float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                                * 4);
                        canvas.drawText(
                                "Put your hands closer together",
                                classificationX,
                                classificationY,
                                betterFormPaint);
                    }
                }

                if (leftNeckAngle < 110f || rightNeckAngle < 110f) {
                    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                    float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                            * 5);
                    canvas.drawText(
                            "Keep your head straight",
                            classificationX,
                            classificationY,
                            betterFormPaint);
                }

                if (Math.abs(leftKneeAngle-rightKneeAngle) > 20f) {
                    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                    float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                            * 6);
                    canvas.drawText(
                            "Don't cave your knees in",
                            classificationX,
                            classificationY,
                            betterFormPaint);
                }

            }

            if (poseClassification.get(0).contains("sumo")) {
                if (poseClassification.get(0).contains("sumodeadlift_down")) {
                    if (leftArmpitAngle > 35f || rightArmpitAngle > 35f) {
                        float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                        float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                                * 3);
                        canvas.drawText(
                                "Grip the bar closer together",
                                classificationX,
                                classificationY,
                                betterFormPaint);
                    }

                    if (leftArmpitAngle < 5f || rightArmpitAngle < 5f) {
                        float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                        float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                                * 4);
                        canvas.drawText(
                                "Grip the bar further apart",
                                classificationX,
                                classificationY,
                                betterFormPaint);
                    }

                    if (leftElbowAngle < 160f || rightElbowAngle < 160f) {
                        float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                        float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                                * 5);
                        canvas.drawText(
                                "Keep your arm straight and don't bend it",
                                classificationX,
                                classificationY,
                                betterFormPaint);
                    }
                }

                if (poseClassification.get(0).contains("sumodeadlift_up")) {
                    if (leftKneeAngle < 160f || rightKneeAngle < 160f) {
                        float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                        float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                                * 6);
                        canvas.drawText(
                                "Lock out your legs",
                                classificationX,
                                classificationY,
                                betterFormPaint);
                    }
                }
                if (leftNeckAngle < 110f || rightNeckAngle < 110f) {
                    float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
                    float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                            * 7);
                    canvas.drawText(
                            "Keep your head straight",
                            classificationX,
                            classificationY,
                            betterFormPaint);
                }

            }

        }
    }


    void drawPoint(Canvas canvas, PoseLandmark landmark, Paint paint) {
        PointF3D point = landmark.getPosition3D();
        maybeUpdatePaintColor(paint, canvas, point.getZ());
        canvas.drawCircle(translateX(point.getX()), translateY(point.getY()), DOT_RADIUS, paint);
    }

    void drawLine(Canvas canvas, PoseLandmark startLandmark, PoseLandmark endLandmark, Paint paint) {
        PointF3D start = startLandmark.getPosition3D();
        PointF3D end = endLandmark.getPosition3D();

        // Gets average z for the current body line
        float avgZInImagePixel = (start.getZ() + end.getZ()) / 2;
        maybeUpdatePaintColor(paint, canvas, avgZInImagePixel);

        canvas.drawLine(
                translateX(start.getX()),
                translateY(start.getY()),
                translateX(end.getX()),
                translateY(end.getY()),
                paint);
    }

    private void maybeUpdatePaintColor(Paint paint, Canvas canvas, float zInImagePixel) {
        if (!visualizeZ) {
            return;
        }

        // When visualizeZ is true, sets up the paint to different colors based on z values.
        // Gets the range of z value.
        float zLowerBoundInScreenPixel;
        float zUpperBoundInScreenPixel;

        if (rescaleZForVisualization) {
            zLowerBoundInScreenPixel = min(-0.001f, scale(zMin));
            zUpperBoundInScreenPixel = max(0.001f, scale(zMax));
        } else {
            // By default, assume the range of z value in screen pixel is [-canvasWidth, canvasWidth].
            float defaultRangeFactor = 1f;
            zLowerBoundInScreenPixel = -defaultRangeFactor * canvas.getWidth();
            zUpperBoundInScreenPixel = defaultRangeFactor * canvas.getWidth();
        }

        float zInScreenPixel = scale(zInImagePixel);

        if (zInScreenPixel < 0) {
            // Sets up the paint to draw the body line in red if it is in front of the z origin.
            // Maps values within [zLowerBoundInScreenPixel, 0) to [255, 0) and use it to control the
            // color. The larger the value is, the more red it will be.
            int v = (int) (zInScreenPixel / zLowerBoundInScreenPixel * 255);
            v = Ints.constrainToRange(v, 0, 255);
            paint.setARGB(255, 255, 255 - v, 255 - v);
        } else {
            // Sets up the paint to draw the body line in blue if it is behind the z origin.
            // Maps values within [0, zUpperBoundInScreenPixel] to [0, 255] and use it to control the
            // color. The larger the value is, the more blue it will be.
            int v = (int) (zInScreenPixel / zUpperBoundInScreenPixel * 255);
            v = Ints.constrainToRange(v, 0, 255);
            paint.setARGB(255, 255 - v, 255 - v, 255);
        }
    }
}
