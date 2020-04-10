package de.motis_project.app2.ppr.route;

import android.graphics.Typeface;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.StyleSpan;
import android.widget.TextView;

import com.google.android.gms.maps.model.LatLng;

import java.util.ArrayList;
import java.util.List;

import de.motis_project.app2.R;
import de.motis_project.app2.ppr.MapUtil;
import motis.ppr.CrossingType;
import motis.ppr.Route;
import motis.ppr.RouteStep;
import motis.ppr.RouteStepType;
import motis.ppr.StreetType;

public class StepListBuilder {
    private Route route;
    private ArrayList<StepInfo> stepInfos;

    private int stepType;
    private int streetType;
    private int crossingType;
    private String streetName;
    private double distance;
    private double duration;
    private double accessibility;
    private List<LatLng> path;
    private int icon;

    private static final StyleSpan boldStyle = new StyleSpan(Typeface.BOLD);

    public StepListBuilder(Route route) {
        this.route = route;
    }

    public List<StepInfo> build() {
        stepInfos = new ArrayList<>(route.stepsLength());

        reset();
        for (int i = 0; i < route.stepsLength(); i++) {
            RouteStep step = route.steps(i);
            if (!extendCurrentStep(step)) {
                finishStep();
                startStep(step);
            }
        }
        finishStep();

        return stepInfos;
    }

    private void reset() {
        stepType = RouteStepType.INVALID;
        streetType = StreetType.NONE;
        crossingType = CrossingType.NONE;
        streetName = "";
        distance = 0;
        duration = 0;
        accessibility = 0;
        path = new ArrayList<>();
        icon = R.drawable.ic_directions_walk_black_24dp;
    }

    private void startStep(RouteStep next) {
        stepType = next.stepType();
        streetType = next.streetType();
        crossingType = next.crossingType();
        streetName = next.streetName();
        if (streetName == null) {
            streetName = "";
        }
        distance = next.distance();
        duration = next.duration();
        accessibility = next.accessibility();
        path = new ArrayList<>(next.path().coordinatesLength() / 2);
        MapUtil.appendPath(path, next);
        icon = getIcon(next);
    }

    private boolean extendCurrentStep(RouteStep next) {
        if (!isExtendableStepType(stepType, streetType)) {
            return false;
        }
        if ((next.stepType() == stepType && streetName.equals(next.streetName())
                ||
                (next.stepType() == RouteStepType.CROSSING
                        && next.streetType() == StreetType.SERVICE &&
                        next.streetName().isEmpty()))) {
            distance += next.distance();
            duration += next.duration();
            accessibility += next.accessibility();
            MapUtil.appendPath(path, next);
            return true;
        }
        return false;
    }

    private void finishStep() {
        if (stepType != RouteStepType.INVALID) {
            stepInfos.add(new StepInfo(
                    stepType, streetType, crossingType,
                    distance, duration, accessibility,
                    buildText(),
                    TextView.BufferType.SPANNABLE, path, icon));
        }
        stepType = RouteStepType.INVALID;
    }

    private boolean isExtendableStepType(int step, int street) {
        return step == RouteStepType.STREET
                || (step == RouteStepType.FOOTWAY && !isSpecialStreetType(street));
    }

    private boolean isSpecialStreetType(int type) {
        return type == StreetType.STAIRS
                || type == StreetType.ESCALATOR
                || type == StreetType.MOVING_WALKWAY;
    }

    private void append(SpannableStringBuilder ssb, CharSequence text, Object what, int flags) {
        int start = ssb.length();
        ssb.append(text);
        ssb.setSpan(what, start, ssb.length(), flags);
    }

    private void appendPlain(SpannableStringBuilder ssb, CharSequence text) {
        ssb.append(text);
    }

    private void appendBold(SpannableStringBuilder ssb, CharSequence text) {
        append(ssb, text, boldStyle, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
    }

    private SpannableStringBuilder buildText() {
        SpannableStringBuilder ssb = new SpannableStringBuilder();
        switch (stepType) {
            case RouteStepType.STREET:
                if (streetName.isEmpty()) {
                    appendBold(ssb, "Straße");
                } else {
                    appendBold(ssb, streetName);
                }
                break;
            case RouteStepType.FOOTWAY:
                switch (streetType) {
                    case StreetType.STAIRS:
                        appendBold(ssb, "Treppe");
                        break;
                    case StreetType.ESCALATOR:
                        appendBold(ssb, "Rolltreppe");
                        break;
                    case StreetType.MOVING_WALKWAY:
                        appendBold(ssb, "Fahrsteig");
                        break;
                    default:
                        if (streetName.isEmpty()) {
                            appendBold(ssb, "Fußweg");
                        } else {
                            appendBold(ssb, streetName);
                            appendPlain(ssb, " (Fußweg)");
                        }
                        break;
                }
                break;
            case RouteStepType.CROSSING:
                buildCrossingText(ssb);
                break;
            case RouteStepType.ELEVATOR:
                appendBold(ssb, "Aufzug");
                break;
            default:
                appendPlain(ssb, "-");
                break;
        }
        return ssb;
    }

    private void buildCrossingText(SpannableStringBuilder ssb) {
        if (streetName.isEmpty() || streetName.equals("LINKED CROSSING")) {
            switch (streetType) {
                case StreetType.RAIL:
                    appendPlain(ssb, "Die Gleise");
                    break;
                case StreetType.TRAM:
                    appendPlain(ssb, "Die Straßenbahngleise");
                    break;
                default:
                    appendPlain(ssb, "Die Straße");
                    break;
            }
        } else {
            if (streetName.contains(";")) {
                appendBold(ssb, streetName.replace(";", " / "));
            } else {
                appendBold(ssb, streetName);
            }
        }

        switch (crossingType) {
            case CrossingType.MARKED:
                appendPlain(ssb, " am Zebrastreifen");
                break;
            case CrossingType.SIGNALS:
                appendPlain(ssb, " an der Ampel");
                break;
            case CrossingType.ISLAND:
                appendPlain(ssb, " an der Verkehrsinsel");
                break;
        }

        appendPlain(ssb, " überqueren");
    }

    private int getIcon(RouteStep step) {
        switch (step.stepType()) {
            case RouteStepType.FOOTWAY:
                switch (step.streetType()) {
                    case StreetType.STAIRS:
                        if (step.inclineUp()) {
                            return R.drawable.ic_stairs_up;
                        } else {
                            return R.drawable.ic_stairs_down;
                        }
                    case StreetType.ESCALATOR:
                        if (step.inclineUp()) {
                            return R.drawable.ic_escalator_up;
                        } else {
                            return R.drawable.ic_escalator_down;
                        }
                    default:
                        break;
                }
                break;
            case RouteStepType.ELEVATOR:
                return R.drawable.ic_elevator;
            case RouteStepType.CROSSING:
                switch (step.crossingType()) {
                    case CrossingType.UNMARKED:
                        return R.drawable.ic_crossing_unmarked;
                    case CrossingType.MARKED:
                        return R.drawable.ic_crossing_marked;
                    case CrossingType.SIGNALS:
                        return R.drawable.ic_crossing_lights;
                    case CrossingType.ISLAND:
                        return R.drawable.ic_crossing_island;
                    default:
                        break;
                }
                break;
            default:
                break;
        }
        return R.drawable.ic_directions_walk_black_24dp;
    }
}
