package de.motis_project.app2.intermodal.journey;

import android.support.annotation.Nullable;

import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.ppr.NamedLocation;
import de.motis_project.app2.ppr.profiles.PprSearchOptions;
import motis.Connection;
import motis.Station;
import motis.Stop;
import motis.Walk;

public class WalkUtil {
    @Nullable
    public static WalkKey getLeadingWalkKey(Connection c, PprSearchOptions pprSearchOptions) {
        Walk w = JourneyUtil.getLeadingWalk(c);
        if (w != null) {
            return getWalkKey(w, c.stops(0), c.stops(1), pprSearchOptions);
        }
        return null;
    }

    @Nullable
    public static WalkKey getTrailingWalkKey(Connection c, PprSearchOptions pprSearchOptions) {
        Walk w = JourneyUtil.getTrailingWalk(c);
        if (w != null) {
            return getWalkKey(w, c.stops(c.stopsLength() - 2), c.stops(c.stopsLength() - 1), pprSearchOptions);
        } else {
            return null;
        }
    }

    public static WalkKey getWalkKey(Walk w, Stop fromStop, Stop toStop, PprSearchOptions pprSearchOptions) {
        long duration = (toStop.arrival().time() - fromStop.departure().time()) / 60;
        NamedLocation from = stationToNamedLocation(fromStop.station());
        NamedLocation to = stationToNamedLocation(toStop.station());
        return new WalkKey(from, to, pprSearchOptions, duration, w.accessibility(), w.mumoType());
    }

    public static NamedLocation stationToNamedLocation(Station station) {
        return new NamedLocation(station.name(), station.pos().lat(), station.pos().lng());
    }
}
