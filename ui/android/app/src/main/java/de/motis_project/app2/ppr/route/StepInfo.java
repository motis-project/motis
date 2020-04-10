package de.motis_project.app2.ppr.route;

import android.widget.TextView;

import com.google.android.gms.maps.model.LatLng;

import java.util.List;

public class StepInfo {
    private final int stepType;
    private final int streetType;
    private final int crossingType;
    private final double distance;
    private final double duration;
    private final double accessibility;
    private final CharSequence text;
    private final TextView.BufferType textBufferType;
    private final List<LatLng> path;
    private final int icon;

    public StepInfo(int stepType, int streetType, int crossingType,
                    double distance, double duration, double accessibility,
                    CharSequence text, TextView.BufferType textBufferType,
                    List<LatLng> path, int icon) {
        this.stepType = stepType;
        this.streetType = streetType;
        this.crossingType = crossingType;
        this.distance = distance;
        this.duration = duration;
        this.accessibility = accessibility;
        this.text = text;
        this.textBufferType = textBufferType;
        this.path = path;
        this.icon = icon;
    }

    public int getStepType() {
        return stepType;
    }

    public int getStreetType() {
        return streetType;
    }

    public int getCrossingType() {
        return crossingType;
    }

    public double getDistance() {
        return distance;
    }

    public double getDuration() {
        return duration;
    }

    public double getAccessibility() {
        return accessibility;
    }

    public CharSequence getText() {
        return text;
    }

    public TextView.BufferType getTextBufferType() {
        return textBufferType;
    }

    public List<LatLng> getPath() {
        return path;
    }

    public int getIcon() {
        return icon;
    }
}
