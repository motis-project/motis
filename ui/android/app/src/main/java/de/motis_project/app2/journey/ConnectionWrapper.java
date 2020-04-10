package de.motis_project.app2.journey;

import android.content.Context;
import android.support.annotation.Nullable;
import android.support.v4.content.ContextCompat;

import com.google.android.gms.maps.model.LatLng;

import java.util.ArrayList;

import de.motis_project.app2.R;
import de.motis_project.app2.intermodal.journey.WalkCache;
import de.motis_project.app2.intermodal.journey.WalkKey;
import de.motis_project.app2.intermodal.journey.WalkUtil;
import de.motis_project.app2.ppr.MapUtil;
import de.motis_project.app2.ppr.profiles.PprSearchOptions;
import de.motis_project.app2.ppr.route.RouteWrapper;
import motis.Connection;
import motis.Stop;

public class ConnectionWrapper {
    private static final ArrayList<Integer> COLORS = new ArrayList<>();

    static {
        COLORS.add(R.color.md_blue500);
        COLORS.add(R.color.md_red500);
        COLORS.add(R.color.md_green500);
        COLORS.add(R.color.md_orange500);
        COLORS.add(R.color.md_purple500);
        COLORS.add(R.color.md_pink500);
        COLORS.add(R.color.md_indigo500);
        COLORS.add(R.color.md_teal500);
        COLORS.add(R.color.md_yellow500);
        COLORS.add(R.color.md_brown500);
    }

    private final Connection connection;
    private final int id;
    private final PprSearchOptions pprSearchOptions;
    private String startNameOverride;
    private String destinationNameOverride;
    private boolean intermodal;

    public ConnectionWrapper(Connection connection, int id, PprSearchOptions pprSearchOptions) {
        this.connection = connection;
        this.id = id;
        this.pprSearchOptions = pprSearchOptions;
        this.intermodal = (pprSearchOptions != null);
    }

    public Connection getConnection() {
        return connection;
    }

    public int getId() {
        return id;
    }

    public int getColor(Context context) {
        int idx = id % COLORS.size();
        if (idx < 0) {
            idx += COLORS.size();
        }
        int colorId = COLORS.get(idx);
        return ContextCompat.getColor(context, colorId);
    }

    public Stop getFirstStop() {
        return connection.stops(0);
    }

    public Stop getLastStop() {
        return connection.stops(connection.stopsLength() - 1);
    }

    public String getStartName() {
        if (startNameOverride != null) {
            return startNameOverride;
        } else {
            return getFirstStop().station().name();
        }
    }

    public String getDestinationName() {
        if (destinationNameOverride != null) {
            return destinationNameOverride;
        } else {
            return getLastStop().station().name();
        }
    }

    public LatLng getStartLatLng() {
        return MapUtil.toLatLng(getFirstStop().station().pos());
    }

    public LatLng getDestinationLatLng() {
        return MapUtil.toLatLng(getLastStop().station().pos());
    }

    public boolean isIntermodal() {
        return intermodal;
    }

    public void setStartNameOverride(String startNameOverride) {
        this.startNameOverride = startNameOverride;
    }

    public void setDestinationNameOverride(String destinationNameOverride) {
        this.destinationNameOverride = destinationNameOverride;
    }

    @Nullable
    public RouteWrapper getLeadingWalkRoute() {
        WalkKey leadingWalkKey = getLeadingWalkKey();
        return (leadingWalkKey != null) ? WalkCache.getInstance().get(leadingWalkKey) : null;
    }

    public WalkKey getLeadingWalkKey() {
        return WalkUtil.getLeadingWalkKey(connection, pprSearchOptions);
    }

    @Nullable
    public RouteWrapper getTrailingWalkRoute() {
        WalkKey trailingWalkKey = getTrailingWalkKey();
        return (trailingWalkKey != null) ? WalkCache.getInstance().get(trailingWalkKey) : null;
    }

    public WalkKey getTrailingWalkKey() {
        return WalkUtil.getTrailingWalkKey(connection, pprSearchOptions);
    }

    public Iterable<LatLng> getPath() {
        RouteWrapper leadingWalk = getLeadingWalkRoute();
        RouteWrapper trailingWalk = getTrailingWalkRoute();

        ArrayList<LatLng> path = new ArrayList<>(connection.stopsLength());
        int start = (leadingWalk == null) ? 0 : 1;
        int end = (trailingWalk == null) ? connection.stopsLength() : connection.stopsLength() - 1;
        if (leadingWalk != null) {
            path.addAll(leadingWalk.getPath());
        }
        for (int i = start; i < end; i++) {
            path.add(MapUtil.toLatLng(connection.stops(i).station().pos()));
        }
        if (trailingWalk != null) {
            path.addAll(trailingWalk.getPath());
        }
        return path;
    }

    public int getNumberOfTransfers() {
        int transfers = 0;
        for (int i = 0; i < connection.stopsLength(); i++) {
            if (connection.stops(i).exit()) {
                transfers++;
            }
        }
        return transfers - 1;
    }

    @Override
    public String toString() {
        return "ConnectionWrapper{" +
                "connection=" + connection +
                ", id=" + id +
                ", pprSearchOptions=" + pprSearchOptions +
                ", startNameOverride='" + startNameOverride + '\'' +
                ", destinationNameOverride='" + destinationNameOverride + '\'' +
                '}';
    }
}
