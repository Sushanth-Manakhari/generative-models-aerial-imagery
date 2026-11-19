"""Part 6 (ROOF-305) helpers: damage labels, the few-label probe, the triage table and the inspection queue."""
import pytest
import torch

from conftest import make_squares
from genroofs import risk
from genroofs.models import RoofAE, RoofVAE


def manifest_of(levels, places=None):
    places = places or ["Alpha"] * len(levels)
    return [{"chip_id": f"{i + 1:06d}", "place": p, "damage": d, "assessment": ""}
            for i, (d, p) in enumerate(zip(levels, places))]


def test_damaged_reads_fema_levels():
    m = manifest_of(["Affected", "none recorded", "Destroyed", "No Damage", "Major", "Unknown"])
    assert risk.damaged(m).tolist() == [True, False, True, False, True, False]
    with pytest.raises(ValueError, match="storm pack"):
        risk.damaged([{"chip_id": "000001", "place": "Alpha"}])


def test_spread_picks_evenly():
    mask = torch.tensor([True, False] * 10)
    assert risk.spread(mask, 4).tolist() == [0, 6, 12, 18]
    assert risk.spread(mask, 50).tolist() == list(range(0, 20, 2))
    assert len(risk.spread(torch.zeros(5, dtype=torch.bool), 3)) == 0


@pytest.mark.parametrize("cls", [RoofAE, RoofVAE])
def test_middle_numbers_are_the_bottleneck(cls):
    torch.manual_seed(0)
    model = cls(latent=16).train()
    imgs = make_squares(10)
    z = risk.middle_numbers(model, imgs, batch_size=4)
    assert z.shape == (10, 16) and z.device.type == "cpu" and model.training
    again = risk.middle_numbers(model, imgs)
    assert torch.allclose(z, again)                      # no random part, even for the VAE


def test_labelled_split_is_balanced_and_disjoint():
    labels = torch.tensor([True] * 30 + [False] * 70)
    learn, test = risk.labelled_split(labels, per_class=10, seed=3)
    assert len(learn) == 20 and int(labels[learn].sum()) == 10
    assert len(test) == 80 and not set(learn.tolist()) & set(test.tolist())
    assert torch.equal(learn, risk.labelled_split(labels, per_class=10, seed=3)[0])
    with pytest.raises(ValueError, match="smaller per_class"):
        risk.labelled_split(labels, per_class=30)


def test_probe_learns_a_simple_rule():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(400, 8, generator=g)
    y = (x[:, 2] - 0.5 * x[:, 5] + 0.3 * torch.randn(400, generator=g)) > 0
    probe = risk.fit_probe(x[:100], y[:100])
    p = probe.predict(x[100:])
    assert p.shape == (300,) and 0 <= float(p.min()) and float(p.max()) <= 1
    out = risk.triage_table(p, y[100:], ks=(50,), show=False)
    assert out["auc"] > 0.9 and out["rows"][0]["precision"] > 0.85
    with pytest.raises(ValueError):
        risk.fit_probe(x[:10], torch.zeros(10, dtype=torch.bool))


def test_precision_recall_and_triage_table(capsys):
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    labels = [1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    assert risk.precision_at_k(scores, labels, 2) == 1.0
    assert risk.precision_at_k(scores, labels, 4) == 0.75
    assert risk.recall_at_k(scores, labels, 4) == 0.75
    out = risk.triage_table(scores, labels, ks=(2, 4), name="toy")
    assert out["base_rate"] == pytest.approx(0.4) and out["rows"][1]["lift"] == pytest.approx(0.75 / 0.4)
    assert out["auc"] == pytest.approx(18 / 24)          # 18 of the 4 x 6 damaged-vs-undamaged pairs
    text = capsys.readouterr().out
    assert "toy: 10 roofs, 4 with damage recorded (base rate 40%)" in text and "AUC 0.75" in text


def test_share_by_place(capsys):
    m = manifest_of(["Affected", "none recorded", "Destroyed", "none recorded"], ["B", "B", "A", "A"])
    rows = risk.share_by_place([0.8, 0.4, 0.9, 0.1], m, risk.damaged(m))
    assert [r["place"] for r in rows] == ["A", "B"]
    assert rows[0]["expected"] == pytest.approx(0.5) and rows[1]["fema"] == pytest.approx(0.5)
    assert "model expects" in capsys.readouterr().out


def test_show_queue_draws_the_top_roofs_in_order():
    imgs = make_squares(20)
    scores = torch.linspace(0, 1, 20)
    labels = torch.arange(20) % 3 == 0
    fig = risk.show_queue(imgs, labels, scores, k=8, ncol=4, show=False)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles[0].startswith("#1 · 1.00") and len(titles) == 8
    assert sum("damage" in t for t in titles) == int(labels[12:].sum())
    assert "first 8 visits" in fig._suptitle.get_text()


def test_random_split_is_disjoint_and_repeatable():
    learn, test = risk.random_split(100, 0.3, seed=4)
    assert len(test) == 30 and len(learn) == 70 and not set(learn.tolist()) & set(test.tolist())
    assert torch.equal(test, risk.random_split(100, 0.3, seed=4)[1])


def _bright_patch_roofs(n=96, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(40, 120, (n, 3, size, size), generator=g, dtype=torch.uint8)
    labels = torch.arange(n) % 2 == 0
    for i in torch.nonzero(labels).flatten().tolist():
        y, x = torch.randint(0, size - 8, (2,), generator=g).tolist()
        imgs[i, :, y:y + 8, x:x + 8] = 240                        # "damage": a bright patch somewhere on the roof
    return imgs, labels


def test_damage_model_learns_an_easy_rule_and_scores_in_0_1(capsys):
    imgs, labels = _bright_patch_roofs()
    net = risk.train_damage_model(imgs, labels, epochs=10, batch_size=16, lr=3e-3, device="cpu")
    assert "pass 10/10" in capsys.readouterr().out and not net.training
    test_imgs, test_labels = _bright_patch_roofs(seed=1)
    p = risk.damage_scores(net, test_imgs, batch_size=40)
    assert p.shape == (96,) and 0 <= float(p.min()) and float(p.max()) <= 1
    assert risk.triage_table(p, test_labels, ks=(10,), show=False)["auc"] > 0.9
    assert risk.DamageNet()(torch.zeros(2, 3, 128, 128)).shape == (2,)       # any roof size works
    with pytest.raises(ValueError):
        risk.train_damage_model(imgs[:, 0], labels)


def test_quiz_roofs_and_reveal(capsys):
    labels = torch.tensor([True] * 10 + [False] * 30)
    quiz = risk.quiz_roofs(labels, 12, seed=2)
    assert len(quiz) == 12 and int(labels[quiz].sum()) == 6 and len(set(quiz.tolist())) == 12
    assert torch.equal(quiz, risk.quiz_roofs(labels, 12, seed=2))
    rows = risk.reveal(quiz, labels, torch.linspace(0, 1, 40))
    assert [r["roof"] for r in rows] == list(range(1, 13)) and {r["fema"] for r in rows} == {"damage recorded",
                                                                                              "none recorded"}
    assert "model's chance of damage" in capsys.readouterr().out
