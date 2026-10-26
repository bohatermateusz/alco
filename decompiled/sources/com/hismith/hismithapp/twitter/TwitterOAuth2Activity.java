package com.hismith.hismithapp.twitter;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.OnApplyWindowInsetsListener;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import com.bumptech.glide.load.Key;
import com.facebook.internal.ServerProtocol;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import kotlin.Metadata;
import kotlin.collections.CollectionsKt;
import kotlin.collections.IntIterator;
import kotlin.jvm.functions.Function1;
import kotlin.jvm.internal.Intrinsics;
import kotlin.random.Random;
import kotlin.ranges.CharRange;
import kotlin.ranges.IntRange;
import kotlin.text.StringsKt;

/* compiled from: TwitterOAuth2Activity.kt */
@Metadata(d1 = {"\u00003\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0002\b\u0003\n\u0002\u0010\u000e\n\u0000\n\u0002\u0010 \n\u0002\u0010\f\n\u0002\b\n\n\u0002\u0010\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\b\u0005\n\u0002\b\u0003*\u0001\u001b\u0018\u00002\u00020\u0001B\u0007¢\u0006\u0004\b\u0002\u0010\u0003J\u0012\u0010\u0012\u001a\u00020\u00132\b\u0010\u0014\u001a\u0004\u0018\u00010\u0015H\u0014J6\u0010\u0016\u001a\u00020\u00052\u0006\u0010\u0017\u001a\u00020\u00052\f\u0010\u0018\u001a\b\u0012\u0004\u0012\u00020\u00050\u00072\u0006\u0010\u0004\u001a\u00020\u00052\u0006\u0010\u0019\u001a\u00020\u00052\u0006\u0010\u0010\u001a\u00020\u0005H\u0002R\u000e\u0010\u0004\u001a\u00020\u0005X\u0082D¢\u0006\u0002\n\u0000R\u0017\u0010\u0006\u001a\b\u0012\u0004\u0012\u00020\b0\u0007¢\u0006\b\n\u0000\u001a\u0004\b\t\u0010\nR\u0011\u0010\u000b\u001a\u00020\u0005¢\u0006\b\n\u0000\u001a\u0004\b\f\u0010\rR\u0017\u0010\u000e\u001a\b\u0012\u0004\u0012\u00020\b0\u0007¢\u0006\b\n\u0000\u001a\u0004\b\u000f\u0010\nR\u0011\u0010\u0010\u001a\u00020\u0005¢\u0006\b\n\u0000\u001a\u0004\b\u0011\u0010\rR\u0010\u0010\u001a\u001a\u00020\u001bX\u0082\u0004¢\u0006\u0004\n\u0002\u0010\u001c¨\u0006\u001d"}, d2 = {"Lcom/hismith/hismithapp/twitter/TwitterOAuth2Activity;", "Landroidx/appcompat/app/AppCompatActivity;", "<init>", "()V", "callbackUrl", "", "allowedChars", "", "", "getAllowedChars", "()Ljava/util/List;", "_state", "get_state", "()Ljava/lang/String;", "charPool", "getCharPool", "challenge", "getChallenge", "onCreate", "", "savedInstanceState", "Landroid/os/Bundle;", "generateAuthenticationUrl", "clientId", "scopes", ServerProtocol.DIALOG_PARAM_STATE, "webViewClient", "com/hismith/hismithapp/twitter/TwitterOAuth2Activity$webViewClient$1", "Lcom/hismith/hismithapp/twitter/TwitterOAuth2Activity$webViewClient$1;", "app"}, k = 1, mv = {2, 3, 0}, xi = 48)
/* loaded from: classes3.dex */
public final class TwitterOAuth2Activity extends AppCompatActivity {
    private final String _state;
    private final String challenge;
    private final List<Character> charPool;
    private final TwitterOAuth2Activity$webViewClient$1 webViewClient;
    private final String callbackUrl = Constans.twitter_redirectUri;
    private final List<Character> allowedChars = CollectionsKt.plus((Collection) CollectionsKt.plus((Iterable) new CharRange('A', 'Z'), (Iterable) new CharRange('a', 'z')), (Iterable) new CharRange('0', '9'));

    /* JADX WARN: Type inference failed for: r1v11, types: [com.hismith.hismithapp.twitter.TwitterOAuth2Activity$webViewClient$1] */
    public TwitterOAuth2Activity() {
        IntRange intRange = new IntRange(1, 10);
        ArrayList arrayList = new ArrayList(CollectionsKt.collectionSizeOrDefault(intRange, 10));
        Iterator<Integer> it = intRange.iterator();
        while (it.hasNext()) {
            ((IntIterator) it).nextInt();
            arrayList.add(Character.valueOf(((Character) CollectionsKt.random(this.allowedChars, Random.INSTANCE)).charValue()));
        }
        this._state = CollectionsKt.joinToString$default(arrayList, "", null, null, 0, null, null, 62, null);
        this.charPool = CollectionsKt.plus((Collection<? extends char>) CollectionsKt.plus((Collection<? extends char>) CollectionsKt.plus((Collection<? extends char>) CollectionsKt.plus((Collection<? extends char>) CollectionsKt.plus((Collection) CollectionsKt.plus((Iterable) new CharRange('a', 'z'), (Iterable) new CharRange('A', 'Z')), (Iterable) new CharRange('0', '9')), '-'), '.'), '_'), '~');
        IntRange intRange2 = new IntRange(1, 44);
        ArrayList arrayList2 = new ArrayList(CollectionsKt.collectionSizeOrDefault(intRange2, 10));
        Iterator<Integer> it2 = intRange2.iterator();
        while (it2.hasNext()) {
            ((IntIterator) it2).nextInt();
            arrayList2.add(Integer.valueOf(Random.INSTANCE.nextInt(0, this.charPool.size())));
        }
        ArrayList arrayList3 = arrayList2;
        List<Character> list = this.charPool;
        ArrayList arrayList4 = new ArrayList(CollectionsKt.collectionSizeOrDefault(arrayList3, 10));
        Iterator it3 = arrayList3.iterator();
        while (it3.hasNext()) {
            arrayList4.add(Character.valueOf(list.get(((Number) it3.next()).intValue()).charValue()));
        }
        this.challenge = CollectionsKt.joinToString$default(arrayList4, "", null, null, 0, null, null, 62, null);
        this.webViewClient = new WebViewClient() { // from class: com.hismith.hismithapp.twitter.TwitterOAuth2Activity$webViewClient$1
            @Override // android.webkit.WebViewClient
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                String str;
                StringBuilder sb = new StringBuilder("shouldOverrideUrlLoading url:");
                Intrinsics.checkNotNull(request);
                LogUtil.e(sb.append(request.getUrl()).toString());
                String uri = request.getUrl().toString();
                Intrinsics.checkNotNullExpressionValue(uri, "toString(...)");
                str = TwitterOAuth2Activity.this.callbackUrl;
                if (!StringsKt.startsWith$default(uri, str, false, 2, (Object) null)) {
                    return false;
                }
                String decode = URLDecoder.decode(request.getUrl().toString(), Key.STRING_CHARSET_NAME);
                Intrinsics.checkNotNull(decode);
                if (!StringsKt.contains$default((CharSequence) decode, (CharSequence) "code=", false, 2, (Object) null)) {
                    return true;
                }
                Uri parse = Uri.parse(decode);
                String queryParameter = parse.getQueryParameter("code");
                if (!Intrinsics.areEqual(parse.getQueryParameter(ServerProtocol.DIALOG_PARAM_STATE), TwitterOAuth2Activity.this.get_state())) {
                    TwitterOAuth2Activity.this.setResult(0);
                    TwitterOAuth2Activity.this.finish();
                    return true;
                }
                Intent intent = new Intent();
                intent.putExtra("code", queryParameter);
                intent.putExtra("challenge", TwitterOAuth2Activity.this.getChallenge());
                TwitterOAuth2Activity.this.setResult(-1, intent);
                TwitterOAuth2Activity.this.finish();
                return true;
            }
        };
    }

    private final String generateAuthenticationUrl(String clientId, List<String> scopes, String callbackUrl, String state, String challenge) {
        return "https://x.com/i/oauth2/authorize?response_type=code&client_id=" + clientId + "&redirect_uri=" + callbackUrl + "&state=" + state + "&code_challenge=" + challenge + "&code_challenge_method=plain&scope=" + CollectionsKt.joinToString$default(scopes, "%20", null, null, 0, null, new Function1() { // from class: com.hismith.hismithapp.twitter.TwitterOAuth2Activity$$ExternalSyntheticLambda1
            @Override // kotlin.jvm.functions.Function1
            public final Object invoke(Object obj) {
                return TwitterOAuth2Activity.generateAuthenticationUrl$lambda$0((String) obj);
            }
        }, 30, null);
    }

    static final CharSequence generateAuthenticationUrl$lambda$0(String it) {
        Intrinsics.checkNotNullParameter(it, "it");
        return it;
    }

    static final WindowInsetsCompat onCreate$lambda$0(View v, WindowInsetsCompat insets) {
        Intrinsics.checkNotNullParameter(v, "v");
        Intrinsics.checkNotNullParameter(insets, "insets");
        Insets insets2 = insets.getInsets(WindowInsetsCompat.Type.systemBars());
        Intrinsics.checkNotNullExpressionValue(insets2, "getInsets(...)");
        v.setPadding(insets2.left, insets2.top, insets2.right, insets2.bottom);
        return insets;
    }

    public final List<Character> getAllowedChars() {
        return this.allowedChars;
    }

    public final String getChallenge() {
        return this.challenge;
    }

    public final List<Character> getCharPool() {
        return this.charPool;
    }

    public final String get_state() {
        return this._state;
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable$default(this, null, null, 3, null);
        setContentView(R.layout.activity_twitter_oauth2);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), new OnApplyWindowInsetsListener() { // from class: com.hismith.hismithapp.twitter.TwitterOAuth2Activity$$ExternalSyntheticLambda0
            @Override // androidx.core.view.OnApplyWindowInsetsListener
            public final WindowInsetsCompat onApplyWindowInsets(View view, WindowInsetsCompat windowInsetsCompat) {
                return TwitterOAuth2Activity.onCreate$lambda$0(view, windowInsetsCompat);
            }
        });
        String stringExtra = getIntent().getStringExtra("clientId");
        Intrinsics.checkNotNull(stringExtra);
        String generateAuthenticationUrl = generateAuthenticationUrl(stringExtra, CollectionsKt.listOf((Object[]) new String[]{"offline.access", "tweet.read", "tweet.write", "tweet.moderate.write", "block.read", "block.write", "follows.read", "follows.write", "mute.read", "mute.write", "users.read", "like.write", "like.read"}), this.callbackUrl, this._state, this.challenge);
        View findViewById = findViewById(R.id.webview);
        Intrinsics.checkNotNullExpressionValue(findViewById, "findViewById(...)");
        WebView webView = (WebView) findViewById;
        WebSettings settings = webView.getSettings();
        Intrinsics.checkNotNullExpressionValue(settings, "getSettings(...)");
        settings.setJavaScriptEnabled(true);
        webView.setWebViewClient(this.webViewClient);
        webView.loadUrl(generateAuthenticationUrl);
    }
}
