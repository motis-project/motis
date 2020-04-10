package de.motis_project.app2.ppr;

import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.LatLngBounds;
import com.google.maps.android.SphericalUtil;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import motis.Polyline;
import motis.Position;
import motis.ppr.Route;
import motis.ppr.RouteStep;

public class MapUtil {
    // https://stackoverflow.com/a/31029389
    public static LatLngBounds toBounds(LatLng center, double radiusInMeters) {
        double distanceFromCenterToCorner = radiusInMeters * Math.sqrt(2.0);
        LatLng southwestCorner =
                SphericalUtil.computeOffset(center, distanceFromCenterToCorner, 225.0);
        LatLng northeastCorner =
                SphericalUtil.computeOffset(center, distanceFromCenterToCorner, 45.0);
        return new LatLngBounds(southwestCorner, northeastCorner);
    }


    public static LatLng toLatLng(Position pos) {
        return new LatLng(pos.lat(), pos.lng());
    }

    public static Collection<LatLng> getRoutePath(Route route) {
        Polyline polyline = route.path();
        ArrayList<LatLng> path = new ArrayList<>(polyline.coordinatesLength() / 2);
        for (int i = 0; i < polyline.coordinatesLength() - 1; i+= 2) {
            path.add(new LatLng(polyline.coordinates(i), polyline.coordinates(i + 1)));
        }
        return path;
    }

    public static LatLngBounds getBounds(Iterable<LatLng> path) {
        LatLngBounds.Builder builder = LatLngBounds.builder();
        for (LatLng latLng : path) {
            builder.include(latLng);
        }
        return builder.build();
    }

    public static void appendPath(List<LatLng> out, RouteStep step) {
        Polyline polyline = step.path();
        for (int i = 0; i < polyline.coordinatesLength() -1; i+=2) {
            out.add(new LatLng(polyline.coordinates(i), polyline.coordinates(i + 1)));
        }
    }
}
