package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.PlayLogsModel;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import java.util.List;
import kotlin.Metadata;
import kotlin.jvm.internal.Intrinsics;

/* compiled from: PlayStatisticsAdapter.kt */
@Metadata(d1 = {"\u00006\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0010 \n\u0002\u0018\u0002\n\u0002\b\u0004\n\u0002\u0018\u0002\n\u0000\n\u0002\u0010\b\n\u0002\b\u0002\n\u0002\u0010\u0002\n\u0002\b\u0004\u0018\u00002\f\u0012\b\u0012\u00060\u0002R\u00020\u00000\u0001:\u0001\u0014B\u001d\u0012\u0006\u0010\u0003\u001a\u00020\u0004\u0012\f\u0010\u0005\u001a\b\u0012\u0004\u0012\u00020\u00070\u0006¢\u0006\u0004\b\b\u0010\tJ\u001c\u0010\n\u001a\u00060\u0002R\u00020\u00002\u0006\u0010\u000b\u001a\u00020\f2\u0006\u0010\r\u001a\u00020\u000eH\u0016J\b\u0010\u000f\u001a\u00020\u000eH\u0016J\u001c\u0010\u0010\u001a\u00020\u00112\n\u0010\u0012\u001a\u00060\u0002R\u00020\u00002\u0006\u0010\u0013\u001a\u00020\u000eH\u0016R\u000e\u0010\u0003\u001a\u00020\u0004X\u0082\u0004¢\u0006\u0002\n\u0000R\u0014\u0010\u0005\u001a\b\u0012\u0004\u0012\u00020\u00070\u0006X\u0082\u0004¢\u0006\u0002\n\u0000¨\u0006\u0015"}, d2 = {"Lcom/hismith/hismithapp/adapter/PlayStatisticsAdapter;", "Landroidx/recyclerview/widget/RecyclerView$Adapter;", "Lcom/hismith/hismithapp/adapter/PlayStatisticsAdapter$ViewHolder;", "context", "Landroid/content/Context;", "data", "", "Lcom/hismith/hismithapp/model/PlayLogsModel;", "<init>", "(Landroid/content/Context;Ljava/util/List;)V", "onCreateViewHolder", "parent", "Landroid/view/ViewGroup;", "viewType", "", "getItemCount", "onBindViewHolder", "", "holder", "position", "ViewHolder", "app"}, k = 1, mv = {2, 3, 0}, xi = 48)
/* loaded from: classes3.dex */
public final class PlayStatisticsAdapter extends RecyclerView.Adapter<ViewHolder> {
    private final Context context;
    private final List<PlayLogsModel> data;

    /* compiled from: PlayStatisticsAdapter.kt */
    @Metadata(d1 = {"\u0000\"\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\b\u0003\n\u0002\u0018\u0002\n\u0002\b\u0005\n\u0002\u0018\u0002\n\u0002\b\b\b\u0086\u0004\u0018\u00002\u00020\u0001B\u000f\u0012\u0006\u0010\u0002\u001a\u00020\u0003¢\u0006\u0004\b\u0004\u0010\u0005R\u001a\u0010\u0006\u001a\u00020\u0007X\u0086\u000e¢\u0006\u000e\n\u0000\u001a\u0004\b\b\u0010\t\"\u0004\b\n\u0010\u000bR\u001a\u0010\f\u001a\u00020\rX\u0086\u000e¢\u0006\u000e\n\u0000\u001a\u0004\b\u000e\u0010\u000f\"\u0004\b\u0010\u0010\u0011R\u001a\u0010\u0012\u001a\u00020\rX\u0086\u000e¢\u0006\u000e\n\u0000\u001a\u0004\b\u0013\u0010\u000f\"\u0004\b\u0014\u0010\u0011¨\u0006\u0015"}, d2 = {"Lcom/hismith/hismithapp/adapter/PlayStatisticsAdapter$ViewHolder;", "Landroidx/recyclerview/widget/RecyclerView$ViewHolder;", "itemView", "Landroid/view/View;", "<init>", "(Lcom/hismith/hismithapp/adapter/PlayStatisticsAdapter;Landroid/view/View;)V", "img_product", "Landroid/widget/ImageView;", "getImg_product", "()Landroid/widget/ImageView;", "setImg_product", "(Landroid/widget/ImageView;)V", "txt_play_date", "Landroid/widget/TextView;", "getTxt_play_date", "()Landroid/widget/TextView;", "setTxt_play_date", "(Landroid/widget/TextView;)V", "txt_play_time", "getTxt_play_time", "setTxt_play_time", "app"}, k = 1, mv = {2, 3, 0}, xi = 48)
    public final class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView img_product;
        final /* synthetic */ PlayStatisticsAdapter this$0;
        private TextView txt_play_date;
        private TextView txt_play_time;

        /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
        public ViewHolder(PlayStatisticsAdapter playStatisticsAdapter, View itemView) {
            super(itemView);
            Intrinsics.checkNotNullParameter(itemView, "itemView");
            this.this$0 = playStatisticsAdapter;
            View findViewById = itemView.findViewById(R.id.img_product);
            Intrinsics.checkNotNullExpressionValue(findViewById, "findViewById(...)");
            this.img_product = (ImageView) findViewById;
            View findViewById2 = itemView.findViewById(R.id.txt_play_date);
            Intrinsics.checkNotNullExpressionValue(findViewById2, "findViewById(...)");
            this.txt_play_date = (TextView) findViewById2;
            View findViewById3 = itemView.findViewById(R.id.txt_play_time);
            Intrinsics.checkNotNullExpressionValue(findViewById3, "findViewById(...)");
            this.txt_play_time = (TextView) findViewById3;
        }

        public final ImageView getImg_product() {
            return this.img_product;
        }

        public final TextView getTxt_play_date() {
            return this.txt_play_date;
        }

        public final TextView getTxt_play_time() {
            return this.txt_play_time;
        }

        public final void setImg_product(ImageView imageView) {
            Intrinsics.checkNotNullParameter(imageView, "<set-?>");
            this.img_product = imageView;
        }

        public final void setTxt_play_date(TextView textView) {
            Intrinsics.checkNotNullParameter(textView, "<set-?>");
            this.txt_play_date = textView;
        }

        public final void setTxt_play_time(TextView textView) {
            Intrinsics.checkNotNullParameter(textView, "<set-?>");
            this.txt_play_time = textView;
        }
    }

    /* JADX WARN: Multi-variable type inference failed */
    public PlayStatisticsAdapter(Context context, List<? extends PlayLogsModel> data) {
        Intrinsics.checkNotNullParameter(context, "context");
        Intrinsics.checkNotNullParameter(data, "data");
        this.context = context;
        this.data = data;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.data.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder holder, int position) {
        Intrinsics.checkNotNullParameter(holder, "holder");
        Glide.with(this.context).load(this.data.get(position).getProductLogo()).placeholder(R.drawable.bg_circle_1affffff).into(holder.getImg_product());
        holder.getTxt_play_date().setText(TimeUtil.utc2Local(this.data.get(position).getPlayDate()));
        holder.getTxt_play_time().setText(LanguageUtil.getLanguageByKey(this.context, "playTime") + this.data.get(position).getPlayLong());
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        Intrinsics.checkNotNullParameter(parent, "parent");
        View inflate = LayoutInflater.from(this.context).inflate(R.layout.item_play_statistics, parent, false);
        Intrinsics.checkNotNullExpressionValue(inflate, "inflate(...)");
        return new ViewHolder(this, inflate);
    }
}
