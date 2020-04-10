package de.motis_project.app2.intermodal;

import android.support.annotation.Nullable;

import java.util.Date;
import java.util.List;

import de.motis_project.app2.intermodal.journey.WalkCache;
import de.motis_project.app2.intermodal.journey.WalkKey;
import de.motis_project.app2.io.Status;
import de.motis_project.app2.journey.ConnectionLoader;
import de.motis_project.app2.journey.ConnectionWrapper;
import de.motis_project.app2.ppr.profiles.PprSearchOptions;
import rx.Subscription;
import rx.android.schedulers.AndroidSchedulers;
import rx.functions.Action1;
import rx.schedulers.Schedulers;


public class IntermodalConnectionLoader extends ConnectionLoader<IntermodalQuery> {
    public IntermodalConnectionLoader(IntermodalQuery query) {
        setQuery(query);
    }

    @Override
    protected Date getQueryTime() {
        return query.getDateTime().getTime();
    }

    @Override
    protected void route(Date searchIntervalBegin, Date searchIntervalEnd,
                         boolean extendIntervalEarlier, boolean extendIntervalLater,
                         int min_connection_count,
                         Action1 action, Action1<Throwable> errorAction) {
        Subscription sub = Status.get().getServer()
                .intermodalRoute(query,
                        searchIntervalBegin, searchIntervalEnd,
                        extendIntervalEarlier, extendIntervalLater,
                        min_connection_count)
                .subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(action, errorAction);
        subscriptions.add(sub);
    }

    @Nullable
    @Override
    protected PprSearchOptions getPprSearchOptions() {
        return query.getPPRSettings().pprSearchOptions;
    }

    @Nullable
    @Override
    protected String getStartNameOverride() {
        return query.getPlaceFrom().name;
    }

    @Nullable
    @Override
    protected String getDestinationNameOverride() {
        return query.getPlaceTo().name;
    }

    @Override
    protected void newConnectionsLoaded(List<ConnectionWrapper> newConnections) {
        super.newConnectionsLoaded(newConnections);
        WalkCache walkCache = WalkCache.getInstance();
        for (ConnectionWrapper connection : newConnections) {
            WalkKey leadingWalkKey = connection.getLeadingWalkKey();
            WalkKey trailingWalkKey = connection.getTrailingWalkKey();
            if (leadingWalkKey != null) {
                walkCache.getOrRequest(leadingWalkKey, null, null);
            }
            if (trailingWalkKey != null) {
                walkCache.getOrRequest(trailingWalkKey, null, null);
            }
        }
    }
}
