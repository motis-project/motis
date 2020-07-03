package de.motis_project.app2;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.support.annotation.Nullable;
import android.support.v4.content.ContextCompat;
import android.support.v4.graphics.drawable.DrawableCompat;
import android.support.v4.util.LongSparseArray;
import android.view.View;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

import motis.Connection;
import motis.Move;
import motis.MoveWrapper;
import motis.Range;
import motis.Stop;
import motis.Transport;
import motis.Trip;
import motis.Walk;

public class JourneyUtil {
    public static final LongSparseArray<Integer> colors = new LongSparseArray<>();
    public static final int WALK_CLASS = 13;

    static {
        colors.put(0, R.color.class_air);
        colors.put(1, R.color.class_ice);
        colors.put(2, R.color.class_ic);
        colors.put(3, R.color.class_coach);
        colors.put(4, R.color.class_n);
        colors.put(5, R.color.class_re);
        colors.put(6, R.color.class_rb);
        colors.put(7, R.color.class_s);
        colors.put(8, R.color.class_u);
        colors.put(9, R.color.class_str);
        colors.put(10, R.color.class_bus);
        colors.put(11, R.color.class_ship);
        colors.put(12, R.color.class_other);
        colors.put(WALK_CLASS, R.color.class_walk);
    }

    public static final LongSparseArray<Integer> icons = new LongSparseArray<>();

    static {
        icons.put(0, R.drawable.ic_baseline_flight_24);
        icons.put(1, R.drawable.ic_directions_railway_black_24dp);
        icons.put(2, R.drawable.ic_directions_railway_black_24dp);
        icons.put(3, R.drawable.ic_directions_bus_black_24dp);
        icons.put(4, R.drawable.ic_directions_railway_black_24dp);
        icons.put(5, R.drawable.ic_directions_railway_black_24dp);
        icons.put(6, R.drawable.ic_directions_railway_black_24dp);
        icons.put(7, R.drawable.sbahn);
        icons.put(8, R.drawable.ubahn);
        icons.put(9, R.drawable.tram);
        icons.put(10, R.drawable.ic_directions_bus_black_24dp);
        icons.put(11, R.drawable.ic_baseline_directions_boat_24);
        icons.put(12, R.drawable.ic_directions_bus_black_24dp);
        icons.put(WALK_CLASS, R.drawable.walk);
    }

    public static int getColor(Context c, long clasz) {
        int id = colors.get(clasz, R.color.class_other);
        return ContextCompat.getColor(c, id);
    }

    public static int getIcon(Context c, long clasz) {
        return icons.get(clasz, R.drawable.ic_directions_bus_black_24dp);
    }

    public static void tintBackground(Context context, View view, long clasz) {
        Drawable bg = DrawableCompat.wrap(view.getBackground());
        DrawableCompat.setTint(bg.mutate(), JourneyUtil.getColor(context, clasz));
        view.setBackground(bg);
    }

    public static void setBackgroundColor(Context context, View view, long clasz) {
        view.setBackgroundColor(JourneyUtil.getColor(context, clasz));
    }

    public static void setIcon(Context context, TextView view, long clasz) {
        view.setCompoundDrawablesWithIntrinsicBounds(JourneyUtil.getIcon(context, clasz), 0, 0, 0);
    }

    public static class Section {
        public final int from, to;

        public Section(int from, int to) {
            this.from = from;
            this.to = to;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj != null && obj instanceof Section) {
                Section other = (Section) obj;
                return other.from == this.from && other.to == this.to;
            }
            return false;
        }

        @Override
        public int hashCode() {
            return from * 1000 + to;
        }
    }

    public static class DisplayTransport {
        public final long clasz;
        public final String longName;
        public final String shortName;

        public DisplayTransport(Transport t) {
            if (useLineId(t)) {
                longName = t.lineId();
                shortName = t.lineId();
            } else {
                longName = t.name();
                shortName = getShortName(t);
            }
            clasz = t.clasz();
        }

        public DisplayTransport(Walk w) {
            longName = "";
            shortName = "";
            clasz = JourneyUtil.WALK_CLASS;
        }

        private static String getShortName(Transport t) {
            if (t.name().length() < 7) {
                return t.name();
            } else if (t.lineId().isEmpty() && t.trainNr() == 0) {
                return t.name();
            } else if (t.lineId().isEmpty()) {
                return Long.toString(t.trainNr());
            } else {
                return t.lineId();
            }
        }

        private static boolean useLineId(Transport t) {
            return t.clasz() == 5 || t.clasz() == 6;
        }
    }

    public static List<Section> getSections(Connection con, boolean includeIntermodalWalks) {
        return SectionsExtractor.getSections(con, includeIntermodalWalks);
    }

    public static List<DisplayTransport> getTransports(Connection con, boolean includeIntermodalWalks) {
        List<DisplayTransport> displayTransports = new ArrayList<>();
        for (Section s : getSections(con, includeIntermodalWalks)) {
            MoveWrapper m = getMove(con, s);
            Transport t = getTransport(m);
            Walk w = getWalk(m);
            if (t != null) {
                displayTransports.add(new DisplayTransport(t));
            } else if (w != null) {
                displayTransports.add(new DisplayTransport(w));
            }
        }
        return displayTransports;
    }

    @Nullable
    public static MoveWrapper getMove(Connection c, Section s) {
        for (int i = 0; i < c.transportsLength(); i++) {
            MoveWrapper m = c.transports(i);
            if (m.moveType() == Move.Transport) {
                Transport t = new Transport();
                m.move(t);

                if (t.range().from() <= s.to && t.range().to() > s.from) {
                    return m;
                }
            } else if (m.moveType() == Move.Walk) {
                Walk w = new Walk();
                m.move(w);

                if (w.range().from() == s.from && w.range().to() == s.to) {
                    return m;
                }
            }
        }
        return null;
    }

    @Nullable
    public static Transport getTransport(Connection c, Section s) {
        return getTransport(getMove(c, s));
    }

    @Nullable
    public static Walk getWalk(Connection c, Section s) {
        return getWalk(getMove(c, s));
    }

    @Nullable
    public static Transport getTransport(@Nullable MoveWrapper m) {
        if (m != null && m.moveType() == Move.Transport) {
            Transport t = new Transport();
            m.move(t);
            return t;
        }
        return null;
    }

    @Nullable
    public static Walk getWalk(@Nullable MoveWrapper m) {
        if (m != null && m.moveType() == Move.Walk) {
            Walk w = new Walk();
            m.move(w);
            return w;
        }
        return null;
    }

    @Nullable
    public static Walk getLeadingWalk(Connection c) {
        if (c.stopsLength() > 1 && !c.stops(0).enter()) {
          return getWalk(c, new Section(0, 1));
        }
        return null;
    }

    @Nullable
    public static Walk getTrailingWalk(Connection c) {
        if (c.stopsLength() > 1 && !c.stops(c.stopsLength() - 1).exit()) {
            return getWalk(c, new Section(c.stopsLength() - 2, c.stopsLength() - 1));
        }
        return null;
    }

    public static String getTransportName(Connection c, Section s) {
        Transport t = JourneyUtil.getTransport(c, s);
        if (t == null) {
            return "";
        }
        return DisplayTransport.useLineId(t) ? Str.san(t.lineId()) : Str.san(t.name());
    }

    public static void printJourney(Connection con) {
        System.out.println("Stops:");
        for (int stopIdx = 0; stopIdx < con.stopsLength(); ++stopIdx) {
            Stop stop = con.stops(stopIdx);
            System.out.print("  " + stopIdx + ": " + stop.station().name());
            System.out.print(" enter=" + stop.enter());
            System.out.print(" exit=" + stop.exit());
            System.out.println();
        }

        System.out.println("Transports:");
        for (int trIdx = 0; trIdx < con.transportsLength(); ++trIdx) {
            System.out.print("  " + trIdx + ": ");
            switch (con.transports(trIdx).moveType()) {
                case Move.Transport: {
                    Transport tr = new Transport();
                    con.transports(trIdx).move(tr);
                    System.out.print(" from=" + tr.range().from() + ", to=" + tr.range().to());
                    System.out.println(" | transport " + tr.name());
                    break;
                }

                case Move.Walk: {
                    Walk tr = new Walk();
                    con.transports(trIdx).move(tr);
                    System.out.print(" from=" + tr.range().from() + ", to=" + tr.range().to());
                    System.out.println(" | walk " + tr.mumoType());
                    break;
                }
            }
        }

        System.out.println("Trips:");
        for (int trpIdx = 0; trpIdx < con.tripsLength(); ++trpIdx) {
            Trip trp = con.trips(trpIdx);
            Range range = trp.range();
            System.out.println("  trip [" + range.from() + ", " + range.to() + "]: " + trp.id().trainNr());
        }
    }
}
