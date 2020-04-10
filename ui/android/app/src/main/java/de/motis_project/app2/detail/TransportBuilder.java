package de.motis_project.app2.detail;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.ViewGroup;

import java.util.HashSet;
import java.util.List;

import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.intermodal.journey.WalkCache;
import de.motis_project.app2.intermodal.journey.WalkKey;
import de.motis_project.app2.intermodal.journey.WalkUtil;
import de.motis_project.app2.journey.ConnectionWrapper;
import de.motis_project.app2.ppr.profiles.PprSearchOptions;
import de.motis_project.app2.ppr.route.RouteWrapper;
import de.motis_project.app2.ppr.route.StepInfo;
import motis.Connection;
import motis.Move;
import motis.MoveWrapper;
import motis.Stop;
import motis.Transport;
import motis.Walk;

public class TransportBuilder {
    private static final String TAG = "TransportBuilder";

    public static void setConnection(
            LayoutInflater inflater,
            ViewGroup journeyDetails,
            ConnectionWrapper con,
            PprSearchOptions pprSearchOptions,
            HashSet<JourneyUtil.Section> expanded) {
        journeyDetails.removeAllViews();

        JourneyUtil.printJourney(con.getConnection());

        List<JourneyUtil.Section> sections = JourneyUtil.getSections(con.getConnection(), true);
        for (int i = 0; i < sections.size(); i++) {
            boolean isFirst = (i == 0);
            boolean isLast = (i == sections.size() - 1);
            JourneyUtil.Section section = sections.get(i);
            JourneyUtil.Section prevSection = isFirst ? null : sections.get(i - 1);
            boolean expand = expanded.contains(section);
            addMove(inflater, journeyDetails, con, pprSearchOptions, prevSection, section, isFirst, isLast, expand);
        }
    }

    protected static void addMove(
            LayoutInflater inflater,
            ViewGroup journeyDetails,
            ConnectionWrapper con,
            PprSearchOptions pprSearchOptions,
            JourneyUtil.Section prevSection,
            JourneyUtil.Section section,
            boolean isFirst, boolean isLast, boolean expand) {
        MoveWrapper m = JourneyUtil.getMove(con.getConnection(), section);
        if (m.moveType() == Move.Transport) {
            addTransport(
                    inflater, journeyDetails, con, prevSection, section,
                    JourneyUtil.getTransport(m), isFirst, isLast, expand);
        } else if (m.moveType() == Move.Walk) {
            addWalk(
                    inflater, journeyDetails, con, pprSearchOptions, prevSection, section,
                    JourneyUtil.getWalk(m), isFirst, isLast, expand);
        }
    }

    protected static void addTransport(
            LayoutInflater inflater,
            ViewGroup journeyDetails,
            ConnectionWrapper conWrapper,
            JourneyUtil.Section prevSection,
            JourneyUtil.Section section,
            Transport t,
            boolean isFirst, boolean isLast, boolean expand) {
        Connection con = conWrapper.getConnection();
        if (isFirst) {
            journeyDetails.addView(
                    new FirstTransportHeader(con, section, journeyDetails, inflater).getView(), 0);
        } else {
            journeyDetails.addView(
                    new TransportHeader(con, prevSection, section, journeyDetails, inflater).getView());
        }

        journeyDetails.addView(new TransportDetail(conWrapper, section, journeyDetails, inflater).getView());

        journeyDetails.addView(new TransportStops(con, section, journeyDetails, inflater, expand).getView());
        if (expand) {
            for (int i = section.from + 1; i < section.to; i++) {
                journeyDetails.addView(new StopOver(con, section, con.stops(i), journeyDetails, inflater).getView());
            }
        }

        if (isLast) {
            journeyDetails.addView(new FinalArrival(conWrapper, section, journeyDetails, inflater).getView());
        } else {
            journeyDetails.addView(new TransportTargetStation(conWrapper, section, journeyDetails, inflater).getView());
        }
    }

    protected static void addWalk(
            LayoutInflater inflater,
            ViewGroup journeyDetails,
            ConnectionWrapper conWrapper,
            PprSearchOptions pprSearchOptions,
            JourneyUtil.Section prevSection,
            JourneyUtil.Section section,
            Walk w,
            boolean isFirst, boolean isLast, boolean expand) {
        Connection con = conWrapper.getConnection();
        Log.i(TAG, "addWalk: expanded=" + expand);
        journeyDetails.addView(
                new WalkHeader(con, section, w, journeyDetails, inflater).getView());

        journeyDetails.addView(new TransportDetail(conWrapper, section, journeyDetails, inflater).getView());

        journeyDetails.addView(new WalkSummary(con, section, w, journeyDetails, inflater, expand).getView());
        if (expand) {
            Stop fromStop = con.stops(section.from);
            Stop toStop = con.stops(section.to);
            WalkKey walkKey = WalkUtil.getWalkKey(w, fromStop, toStop, pprSearchOptions);
            WalkCache cache = WalkCache.getInstance();
            RouteWrapper route = cache.get(walkKey);
            Log.i(TAG, "Walk: key=" + walkKey + ", route=" + route);
            if (route != null) {
                List<StepInfo> steps = route.getSteps();
                long time = fromStop.departure().time();
                for (StepInfo step : steps) {
                    journeyDetails.addView(new WalkStep(step, time, journeyDetails, inflater).getView());
                    time += (long) step.getDuration();
                }
            } else {
                cache.getOrRequest(walkKey, r -> {
                    DetailClickHandler activity = (DetailClickHandler) inflater.getContext();
                    activity.refreshSection(section);
                }, t -> {
                    Log.w(TAG, "Could not load walk route: " + t);
                    if (t != null) {
                        t.printStackTrace();
                    }
                });
            }
        }

        if (isLast) {
            journeyDetails.addView(new FinalArrival(conWrapper, section, journeyDetails, inflater).getView());
        } else {
            journeyDetails.addView(new TransportTargetStation(conWrapper, section, journeyDetails, inflater).getView());
        }
    }
}
