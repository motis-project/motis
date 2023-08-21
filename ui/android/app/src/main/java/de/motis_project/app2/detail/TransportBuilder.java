package de.motis_project.app2.detail;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.ViewGroup;

import java.util.HashSet;
import java.util.List;

import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.journey.ConnectionWrapper;
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
            addMove(inflater, journeyDetails, con, prevSection, section, isFirst, isLast, expand);
        }
    }

    protected static void addMove(
            LayoutInflater inflater,
            ViewGroup journeyDetails,
            ConnectionWrapper con,
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
                    inflater, journeyDetails, con, prevSection, section,
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

        if (isLast) {
            journeyDetails.addView(new FinalArrival(conWrapper, section, journeyDetails, inflater).getView());
        } else {
            journeyDetails.addView(new TransportTargetStation(conWrapper, section, journeyDetails, inflater).getView());
        }
    }
}
