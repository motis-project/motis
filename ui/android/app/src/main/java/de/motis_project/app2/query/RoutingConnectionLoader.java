package de.motis_project.app2.query;

import androidx.annotation.Nullable;

import java.util.Date;

import de.motis_project.app2.io.Status;
import de.motis_project.app2.journey.ConnectionLoader;
import rx.Subscription;
import rx.android.schedulers.AndroidSchedulers;
import rx.functions.Action1;
import rx.schedulers.Schedulers;


public class RoutingConnectionLoader extends ConnectionLoader<Query> {
    public RoutingConnectionLoader(Query query) {
        setQuery(query);
    }

    @Override
    protected Date getQueryTime() {
        return query.getTime();
    }

    @Override
    protected void route(Date searchIntervalBegin, Date searchIntervalEnd,
                         boolean extendIntervalEarlier, boolean extendIntervalLater,
                         int min_connection_count,
                         Action1 action, Action1<Throwable> errorAction) {
        Subscription sub = Status.get().getServer()
                .route(query.getFromId(), query.getToId(),
                        query.isArrival(),
                        searchIntervalBegin, searchIntervalEnd,
                        extendIntervalEarlier, extendIntervalLater,
                        min_connection_count)
                .subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(action, errorAction);
        subscriptions.add(sub);
    }
}
