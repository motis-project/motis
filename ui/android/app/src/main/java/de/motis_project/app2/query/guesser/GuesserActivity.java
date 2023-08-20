package de.motis_project.app2.query.guesser;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.widget.EditText;
import android.widget.ListView;

import androidx.fragment.app.FragmentActivity;

import com.jakewharton.rxbinding.widget.RxTextView;
import com.jakewharton.rxbinding.widget.TextViewTextChangeEvent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;
import butterknife.OnItemClick;
import de.motis_project.app2.R;
import de.motis_project.app2.io.Status;
import rx.Observable;
import rx.Subscriber;
import rx.Subscription;

public class GuesserActivity extends FragmentActivity {
    public static final String RESULT_NAME = "result_name";
    public static final String RESULT_ID = "result_id";
    public static final String QUERY = "route";

    private GuesserListAdapter adapter;
    private Subscription subscription;

    @BindView(R.id.suggestionslist)
    ListView suggestions;

    @BindView(R.id.searchInput)
    EditText searchInput;

    @OnClick(R.id.backButton)
    void closeActivity() {
        setResult(Activity.RESULT_CANCELED, null);
        finish();
    }

    @OnClick(R.id.clearButton)
    void clearInput() {
        searchInput.setText("");
        adapter.clear();
    }

    @OnItemClick(R.id.suggestionslist)
    void onSuggestionSelected(int pos) {
        StationGuess selected = adapter.getItem(pos);

        Intent i = new Intent();
        i.putExtra(RESULT_NAME, selected.name);
        i.putExtra(RESULT_ID, selected.eva);
        setResult(Activity.RESULT_OK, i);

        Status.get().getFavoritesDb().addOrIncrement(selected.eva, selected.name);

        finish();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.query_guesser_activity);
        ButterKnife.bind(this);

        adapter = new GuesserListAdapter(this);
        suggestions.setAdapter(adapter);

        String query = getIntent().getStringExtra(QUERY);
        if (query != null) {
            subscription = setupSubscription(query);
            searchInput.setText(query);
            searchInput.setSelection(query.length());
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        subscription.unsubscribe();
    }

    @Override
    public void onStart() {
        super.onStart();
    }

    @Override
    public void onStop() {
        super.onStop();
    }

    private Subscription setupSubscription(String init) {
        Observable<List<StationGuess>> favoriteGuesses =
                RxTextView.textChangeEvents(searchInput)
                        .map(TextViewTextChangeEvent::text)
                        .map(CharSequence::toString)
                        .startWith(init)
                        .map(String::toLowerCase)
                        .flatMap(in -> Status.get().getFavoritesDb().getFavorites(in));

        Observable<List<StationGuess>> serverGuesses =
                RxTextView.textChangeEvents(searchInput)
                        .map(TextViewTextChangeEvent::text)
                        .map(CharSequence::toString)
                        .startWith(init)
                        .map(String::toLowerCase)
                        .filter(in -> in.length() >= 3)
                        .flatMap(in -> Status.get().getServer().guess(in))
                        .map(res -> {
                            List<StationGuess> guesses = new ArrayList<>(res.guessesLength());
                            for (int i = 0; i < res.guessesLength(); i++) {
                                guesses.add(new StationGuess(
                                        res.guesses(i).id(),
                                        res.guesses(i).name(),
                                        -i - 1,
                                        StationGuess.SERVER_GUESS));
                            }
                            return guesses;
                        })
                        .startWith(new ArrayList<StationGuess>());

        return Observable
                .combineLatest(favoriteGuesses, serverGuesses, (f, g) -> {
                    final List<StationGuess> guesses = new ArrayList<>();
                    g.removeAll(f);
                    guesses.addAll(f);
                    guesses.addAll(g);
                    Collections.sort(guesses);
                    return guesses;
                })
                .subscribe(new Subscriber<List<StationGuess>>() {
                    @Override
                    public void onCompleted() {
                    }

                    @Override
                    public void onError(Throwable e) {
                    }

                    @Override
                    public void onNext(List<StationGuess> guesses) {
                        runOnUiThread(() -> adapter.setContent(guesses));
                    }
                });
    }
}
