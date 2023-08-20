package de.motis_project.app2.journey;

import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import de.motis_project.app2.JourneyUtil;
import de.motis_project.app2.TimeUtil;
import motis.Connection;
import motis.ProblemType;
import motis.routing.RoutingResponse;
import rx.functions.Action1;
import rx.internal.util.SubscriptionList;

public abstract class ConnectionLoader<Q> {
    private static final String TAG = "ConnectionLoader";
    private static final int SECOND_IN_MS = 1000;
    private static final int MINUTE_IN_MS = 60 * SECOND_IN_MS;
    private static final int HOUR_IN_MS = 60 * MINUTE_IN_MS;
    private static final int SEARCH_INTERVAL_MS = 2 * HOUR_IN_MS;
    private static final int INITIAL_MIN_CONNECTION_COUNT = 5;

    protected Q query;
    protected Date intervalBegin, intervalEnd;

    protected SubscriptionList subscriptions = new SubscriptionList();
    protected final List<ConnectionWrapper> data = new ArrayList<>();
    protected boolean serverError = false;
    protected boolean initialRequestPending = true;
    protected int firstId;
    protected int lastId;

    public void reset() {
        Log.d(TAG, "reset");
        serverError = false;
        initialRequestPending = true;
        data.clear();
        firstId = 0;
        lastId = 0;

        if (query != null) {
            intervalBegin = getQueryTime();
            intervalEnd = new Date(intervalBegin.getTime() + SEARCH_INTERVAL_MS);
        }

        subscriptions.unsubscribe();
        subscriptions = new SubscriptionList();
    }

    public void loadInitial(@Nullable LoadSucceeded success, @Nullable LoadFailed fail) {
        final Date searchIntervalBegin = new Date(intervalBegin.getTime());
        final Date searchIntervalEnd = new Date(intervalEnd.getTime());

        Log.d(TAG, "loadInitial: " + searchIntervalBegin + " - " + searchIntervalEnd);

        route(searchIntervalBegin, searchIntervalEnd, true, true, INITIAL_MIN_CONNECTION_COUNT,
                resObj -> {
                    RoutingResponse res = (RoutingResponse) resObj;
                    Log.d(TAG, "Received initial response");
                    logResponse(res, searchIntervalBegin, searchIntervalEnd, "INITIAL");

                    intervalBegin = new Date(res.intervalBegin() * 1000);
                    intervalEnd = new Date(res.intervalEnd() * 1000);
                    initialRequestPending = false;

                    if (res.connectionsLength() == 0) {
                        Log.i(TAG, "Received response with 0 connections");
                        serverError = true;
                    }

                    List<Connection> conns = getConnections(res);

                    data.clear();
                    firstId = 0;
                    lastId = res.connectionsLength() - 1;
                    for (int i = 0; i < conns.size(); i++) {
                        data.add(wrapConnection(conns.get(i), i));
                    }

                    newConnectionsLoaded(data);

                    if (success != null) {
                        success.loaded(res, 0, data.size());
                    }
                },
                t -> {
                    initialRequestPending = false;
                    serverError = true;
                    Log.w(TAG, "Initial request failed: ", t);
                    if (fail != null) {
                        fail.failed(t);
                    }
                });
    }

    @NonNull
    protected List<Connection> getConnections(RoutingResponse res) {
        List<Connection> conns = new ArrayList<>(res.connectionsLength());
        for (int i = 0; i < res.connectionsLength(); i++) {
            Connection c = res.connections(i);
            if (hasNoProblems(c)) {
                conns.add(c);
            }
        }
        sortConnections(conns);
        return conns;
    }

    protected boolean hasNoProblems(Connection c) {
        for (int i = 0; i < c.problemsLength(); i++) {
            if (c.problems(i).type() != ProblemType.NO_PROBLEM) {
                return false;
            }
        }
        return true;
    }

    public void loadBefore(@Nullable LoadSucceeded success, @Nullable LoadFailed fail) {
        final Date searchIntervalBegin = new Date(intervalBegin.getTime() - SEARCH_INTERVAL_MS);
        final Date searchIntervalEnd = new Date(intervalBegin.getTime() - MINUTE_IN_MS);

        Log.d(TAG, "loadBefore: " + searchIntervalBegin + " - " + searchIntervalEnd);

        route(searchIntervalBegin, searchIntervalEnd, true, false, 0,
                resObj -> {
                    RoutingResponse res = (RoutingResponse) resObj;
                    Log.d(TAG, "Received before response");
                    logResponse(res, searchIntervalBegin, searchIntervalEnd, "LOAD_BEFORE");

                    List<Connection> newConns = getConnections(res);

                    List<ConnectionWrapper> newData = new ArrayList<>(newConns.size());
                    firstId -= newConns.size();
                    for (int i = 0; i < newConns.size(); i++) {
                        newData.add(wrapConnection(newConns.get(i), firstId + i));
                    }
                    newConnectionsLoaded(newData);

                    intervalBegin = searchIntervalBegin;
                    data.addAll(0, newData);

                    if (success != null) {
                        success.loaded(res, 0, newConns.size());
                    }
                },
                t -> {
                    Log.i(TAG, "Before request failed:", t);
                    if (fail != null) {
                        fail.failed(t);
                    }
                });
    }

    public void loadAfter(@Nullable LoadSucceeded success, @Nullable LoadFailed fail) {
        final Date searchIntervalBegin = new Date(intervalEnd.getTime() + MINUTE_IN_MS);
        final Date searchIntervalEnd = new Date(intervalEnd.getTime() + SEARCH_INTERVAL_MS);

        Log.d(TAG, "loadAfter: " + searchIntervalBegin + " - " + searchIntervalEnd);

        route(searchIntervalBegin, searchIntervalEnd, false, true, 0,
                resObj -> {
                    RoutingResponse res = (RoutingResponse) resObj;
                    Log.d(TAG, "Received after response");
                    logResponse(res, searchIntervalBegin, searchIntervalEnd, "LOAD_AFTER");

                    List<Connection> newConns = getConnections(res);

                    List<ConnectionWrapper> newData = new ArrayList<>(newConns.size());
                    for (int i = 0; i < newConns.size(); i++) {
                        newData.add(wrapConnection(newConns.get(i), lastId + i + 1));
                    }
                    lastId += newConns.size();
                    newConnectionsLoaded(newData);

                    intervalEnd = searchIntervalEnd;
                    int oldSize = data.size();
                    data.addAll(newData);

                    if (success != null) {
                        success.loaded(res, oldSize, newData.size());
                    }

                },
                t -> {
                    Log.i(TAG, "After request failed:", t);
                    if (fail != null) {
                        fail.failed(t);
                    }
                });
    }

    protected abstract Date getQueryTime();

    protected abstract void route(Date searchIntervalBegin, Date searchIntervalEnd,
                                  boolean extendIntervalEarlier,
                                  boolean extendIntervalLater,
                                  int min_connection_count,
                                  Action1 action, Action1<Throwable> errorAction);

    @Nullable
    protected String getStartNameOverride() {
        return null;
    }

    @Nullable
    protected String getDestinationNameOverride() {
        return null;
    }

    protected ConnectionWrapper wrapConnection(Connection connection, int id) {
        ConnectionWrapper wrapper = new ConnectionWrapper(connection, id);
        wrapper.setStartNameOverride(getStartNameOverride());
        wrapper.setDestinationNameOverride(getDestinationNameOverride());
        return wrapper;
    }

    protected void sortConnections(List<Connection> data) {
        Collections.sort(data, (a, b) -> {
            long depA = a.stops(0).departure().scheduleTime();
            long depB = b.stops(0).departure().scheduleTime();
            return Long.compare(depA, depB);
        });
    }

    protected void newConnectionsLoaded(List<ConnectionWrapper> newConnections) {

    }

    public void destroy() {
        subscriptions.clear();
    }

    public Q getQuery() {
        return query;
    }

    public void setQuery(Q query) {
        this.query = query;
    }

    public List<ConnectionWrapper> getData() {
        return data;
    }

    public boolean isServerError() {
        return serverError;
    }

    public boolean isInitialRequestPending() {
        return initialRequestPending;
    }

    protected void logResponse(RoutingResponse res, Date intervalBegin, Date intervalEnd,
                               String type) {
        System.out.println(new StringBuilder().append(type).append("  ").append("Routing from ")
                .append(TimeUtil.formatDate(intervalBegin)).append(", ")
                .append(TimeUtil.formatTime(intervalBegin)).append(" until ")
                .append(TimeUtil.formatDate(intervalEnd)).append(", ")
                .append(TimeUtil.formatTime(intervalEnd)));
        for (int i = 0; i < res.connectionsLength(); i++) {
            Connection con = res.connections(i);
            Date depTime = new Date(con.stops(0).departure().scheduleTime() * 1000);
            Date arrTime =
                    new Date(con.stops(con.stopsLength() - 1).arrival().scheduleTime() * 1000);
            int interchangeCount = JourneyUtil.getSections(con, false).size() - 1;
            long travelTime = con.stops(con.stopsLength() - 1).arrival().scheduleTime() -
                    con.stops(0).departure().scheduleTime();

            StringBuilder sb = new StringBuilder();
            sb.append("start: ").append(depTime).append("  ");
            sb.append("end: ").append(arrTime).append("  ");
            sb.append("Duration: ").append(TimeUtil.formatDuration(travelTime / 60))
                    .append("  ");
            sb.append("Interchanges: ").append(interchangeCount).append("  ");
            System.out.println(sb);
        }
    }

    public interface LoadSucceeded {
        public void loaded(RoutingResponse res, int newStartIndex, int newConnectionCount);
    }

    public interface LoadFailed {
        public void failed(Throwable t);
    }
}
