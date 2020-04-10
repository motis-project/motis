package de.motis_project.app2.intermodal.journey;

import android.support.annotation.Nullable;
import android.util.Log;

import java.util.ArrayList;
import java.util.HashMap;

import de.motis_project.app2.io.Status;
import de.motis_project.app2.ppr.query.PPRQuery;
import de.motis_project.app2.ppr.route.RouteWrapper;
import motis.ppr.FootRoutingResponse;
import motis.ppr.Route;
import motis.ppr.Routes;
import rx.Subscription;
import rx.android.schedulers.AndroidSchedulers;
import rx.functions.Action1;
import rx.internal.util.SubscriptionList;
import rx.schedulers.Schedulers;

public class WalkCache {
    private static final String TAG = "WalkCache";
    private static final WalkCache ourInstance = new WalkCache();

    private final HashMap<WalkKey, RouteWrapper> cache = new HashMap<>();
    private final ArrayList<WalkKey> pendingRequests = new ArrayList<>();
    private final SubscriptionList subscriptions = new SubscriptionList();
    private final ArrayList<Listener> listeners = new ArrayList<>();
    private final HashMap<WalkKey, ArrayList<Action1<RouteWrapper>>> queuedSuccessCallbacks = new HashMap<>();
    private final HashMap<WalkKey, ArrayList<Action1<Throwable>>> queuedFailCallbacks = new HashMap<>();

    public static WalkCache getInstance() {
        return ourInstance;
    }

    private WalkCache() {
    }

    private void put(WalkKey key, RouteWrapper value) {
        synchronized (this) {
            cache.put(key, value);
        }
    }

    @Nullable
    public RouteWrapper get(WalkKey key) {
        synchronized (this) {
            return cache.get(key);
        }
    }

    public void getOrRequest(WalkKey key, @Nullable Action1<RouteWrapper> action,
                             @Nullable Action1<Throwable> errorAction) {
        synchronized (this) {
            RouteWrapper route = get(key);
            if (route == null) {
                if (!pendingRequests.contains(key)) {
                    pendingRequests.add(key);
                    request(key, action, errorAction);
                } else {
                    addQueuedCallback(key, action, errorAction);
                }
            } else if (action != null) {
                action.call(route);
            }
        }
    }

    public void addListener(Listener listener) {
        synchronized (listeners) {
            listeners.add(listener);
        }
    }

    public void removeListener(Listener listener) {
        synchronized (listeners) {
            listeners.remove(listener);
        }
    }

    private void request(WalkKey key, @Nullable Action1<RouteWrapper> action,
                         @Nullable Action1<Throwable> errorAction) {
        Log.i(TAG, "request: " + key);
        PPRQuery query = new PPRQuery(null);
        query.placeFrom = key.getFrom();
        query.placeTo = key.getTo();
        query.pprSearchOptions = key.getPprSearchOptions();
        Subscription sub = Status.get().getServer()
                .pprRoute(query)
                .subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(resObj -> {
                    FootRoutingResponse res = (FootRoutingResponse) resObj;
                    Route route = findRoute(res, key);
                    RouteWrapper wrapper =
                            (route != null) ? new RouteWrapper(route, cache.size()) : null;
                    synchronized (this) {
                        pendingRequests.remove(key);
                        if (route != null) {
                            cache.put(key, wrapper);
                        }
                    }
                    Log.i(TAG, "received route for " + key + ": " + wrapper);
                    if (wrapper != null) {
                        if (action != null) {
                            action.call(wrapper);
                        }
                        notifyQueuedSuccessCallbacks(key, wrapper);
                        notifyLoaded(key, wrapper);
                    } else if (errorAction != null) {
                        errorAction.call(null);
                        notifyQueuedFailCallbacks(key, null);
                        notifyRequestFailed(key, null);
                    }
                }, t -> {
                    Log.i(TAG, "request failed for " + key);
                    synchronized (this) {
                        pendingRequests.remove(key);
                    }
                    if (errorAction != null) {
                        errorAction.call(t);
                    }
                    notifyQueuedFailCallbacks(key, t);
                    notifyRequestFailed(key, t);
                });
        subscriptions.add(sub);
    }

    @Nullable
    private Route findRoute(FootRoutingResponse res, WalkKey key) {
        for (int i = 0; i < res.routesLength(); i++) {
            Routes routes = res.routes(i);
            for (int j = 0; j < routes.routesLength(); j++) {
                Route route = routes.routes(j);
                if (route.duration() == key.getDuration()
                        && route.accessibility() == key.getAccessibility()) {
                    return route;
                }
            }
        }
        return null;
    }

    private void notifyLoaded(WalkKey key, RouteWrapper route) {
        synchronized (listeners) {
            for (Listener listener : listeners) {
                listener.routeLoaded(key, route);
            }
        }
    }

    private void notifyRequestFailed(WalkKey key, Throwable t) {
        synchronized (listeners) {
            for (Listener listener : listeners) {
                listener.routeRequestFailed(key, t);
            }
        }
    }

    private void addQueuedCallback(WalkKey key, @Nullable Action1<RouteWrapper> action,
                                   @Nullable Action1<Throwable> errorAction) {
        if (action != null) {
            synchronized (queuedSuccessCallbacks) {
                ArrayList<Action1<RouteWrapper>> callbacks = queuedSuccessCallbacks.get(key);
                if (callbacks == null) {
                    callbacks = new ArrayList<>();
                    queuedSuccessCallbacks.put(key, callbacks);
                }
                callbacks.add(action);
            }
        }
        if (errorAction != null) {
            synchronized (queuedFailCallbacks) {
                ArrayList<Action1<Throwable>> callbacks = queuedFailCallbacks.get(key);
                if (callbacks == null) {
                    callbacks = new ArrayList<>();
                    queuedFailCallbacks.put(key, callbacks);
                }
                callbacks.add(errorAction);
            }
        }
    }

    private void notifyQueuedSuccessCallbacks(WalkKey key, RouteWrapper route) {
        ArrayList<Action1<RouteWrapper>> callbacks = null;
        synchronized (queuedSuccessCallbacks) {
            callbacks = queuedSuccessCallbacks.get(key);
            if (callbacks != null) {
                queuedSuccessCallbacks.remove(key);
            }
        }
        if (callbacks != null) {
            for (Action1<RouteWrapper> action : callbacks) {
                action.call(route);
            }
        }
    }

    private void notifyQueuedFailCallbacks(WalkKey key, Throwable t) {
        ArrayList<Action1<Throwable>> callbacks = null;
        synchronized (queuedFailCallbacks) {
            callbacks = queuedFailCallbacks.get(key);
            if (callbacks != null) {
                queuedFailCallbacks.remove(key);
            }
        }
        if (callbacks != null) {
            for (Action1<Throwable> action : callbacks) {
                action.call(t);
            }
        }
    }

    public interface Listener {
        public void routeLoaded(WalkKey key, RouteWrapper route);

        public void routeRequestFailed(WalkKey key, Throwable t);
    }
}
