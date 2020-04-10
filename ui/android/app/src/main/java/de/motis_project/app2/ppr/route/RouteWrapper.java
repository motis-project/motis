package de.motis_project.app2.ppr.route;

import android.content.Context;
import android.support.v4.content.ContextCompat;

import com.google.android.gms.maps.model.LatLng;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import de.motis_project.app2.R;
import de.motis_project.app2.ppr.MapUtil;
import motis.ppr.Route;
import motis.ppr.RouteStep;
import motis.ppr.RouteStepType;
import motis.ppr.StreetType;

public class RouteWrapper {
    private static final ArrayList<Integer> COLORS = new ArrayList<>();

    static {
        COLORS.add(R.color.md_blue500);
        COLORS.add(R.color.md_red500);
        COLORS.add(R.color.md_green500);
        COLORS.add(R.color.md_orange500);
        COLORS.add(R.color.md_purple500);
        COLORS.add(R.color.md_teal500);
        COLORS.add(R.color.md_yellow500);
        COLORS.add(R.color.md_pink500);
        COLORS.add(R.color.md_brown500);
        COLORS.add(R.color.md_indigo500);
        COLORS.add(R.color.md_deeporange500);
    }

    private final Route route;
    private final int id;

    private int stairCount;
    private int escalatorCount;
    private int elevatorCount;
    private int movingWalkwayCount;

    public RouteWrapper(Route route, int id) {
        this.route = route;
        this.id = id;
        calcStats();
    }

    public Route getRoute() {
        return route;
    }

    public int getId() {
        return id;
    }

    public LatLng getStart() {
        return MapUtil.toLatLng(route.start());
    }

    public LatLng getDestination() {
        return MapUtil.toLatLng(route.destination());
    }

    public Collection<LatLng> getPath() {
        return MapUtil.getRoutePath(route);
    }

    public int getColor(Context context) {
        int colorId = COLORS.get(id % COLORS.size());
        return ContextCompat.getColor(context, colorId);
    }

    public List<StepInfo> getSteps() {
        return new StepListBuilder(route).build();
    }

    public int getStairCount() {
        return stairCount;
    }

    public int getEscalatorCount() {
        return escalatorCount;
    }

    public int getElevatorCount() {
        return elevatorCount;
    }

    public int getMovingWalkwayCount() {
        return movingWalkwayCount;
    }

    private void calcStats() {
        stairCount = 0;
        escalatorCount = 0;
        elevatorCount = 0;
        movingWalkwayCount = 0;
        for (int i = 0; i < route.stepsLength(); i++) {
            RouteStep step = route.steps(i);
            switch (step.stepType()) {
                case RouteStepType.FOOTWAY:
                    switch (step.streetType()) {
                        case StreetType.STAIRS:
                            stairCount++;
                            break;
                        case StreetType.ESCALATOR:
                            escalatorCount++;
                            break;
                        case StreetType.MOVING_WALKWAY:
                            movingWalkwayCount++;
                            break;
                    }
                case RouteStepType.ELEVATOR:
                    elevatorCount++;
                    break;
            }
        }
    }
}
