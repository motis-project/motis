package de.motis_project.app2.query.guesser;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import java.util.List;

import de.motis_project.app2.R;

public class GuesserListAdapter extends ArrayAdapter<StationGuess> {
    private final LayoutInflater inflater;
    private int favoriteCount;

    private static final int FAVORITE_ITEM = 0;
    private static final int SUGGESTED_ITEM = 1;

    public GuesserListAdapter(Context context) {
        super(context, R.layout.query_guesser_list_item);
        inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
    }

    public void setContent(List<StationGuess> suggestions) {
        clear();
        addAll(suggestions);
    }

    @Override
    public int getItemViewType(int position) {
        return getItem(position).type;
    }

    @NonNull
    @Override
    public View getView(int position, View view, ViewGroup parent) {
        if (view == null) {
            view = inflater.inflate(R.layout.query_guesser_list_item, parent, false);
        }

        StationGuess item = getItem(position);

        TextView tv = (TextView) view.findViewById(R.id.guess_text);
        tv.setText(item.name);

        int drawable = item.type == StationGuess.FAVORITE_GUESS
                       ? R.drawable.ic_favorite_black_24dp
                       : R.drawable.ic_place_black_24dp;
        ImageView icon = (ImageView) view.findViewById(R.id.guess_icon);
        icon.setImageDrawable(ContextCompat.getDrawable(getContext(), drawable));

        return view;
    }

    @Override
    public int getViewTypeCount() {
        return 2;
    }
}
