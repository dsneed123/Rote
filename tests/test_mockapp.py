"""The mock app's runtime conditions are what replay is tested against, so
they are pinned here at the HTTP level, without a browser."""

import pytest
import requests

from mockapp.corelink import OPERATOR_PASSWORD, OPERATOR_USER, serve_in_thread


@pytest.fixture
def app():
    server, url = serve_in_thread(port=0, faults=True)
    yield url
    server.shutdown()


def signed_in(url):
    s = requests.Session()
    r = s.post(f"{url}/login", data={"f_0001": OPERATOR_USER, "f_0002": OPERATOR_PASSWORD})
    assert "frameset" in r.text
    return s


def inquire(s, url, number):
    return s.post(f"{url}/inquiry", data={"f_0012": number})


def test_bad_password_is_rejected(app):
    r = requests.post(f"{app}/login", data={"f_0001": OPERATOR_USER, "f_0002": "nope"})
    assert "CL-0101" in r.text


def test_inquiry_reaches_member_detail(app):
    r = inquire(signed_in(app), app, "12345")
    assert "MEMBER INQUIRY - DETAIL" in r.text and "$1,234.56" in r.text


@pytest.mark.parametrize("number,code", [("99999", "CL-0204"), ("12a", "CL-0210"),
                                         ("77777", "CL-0403")])
def test_inquiry_business_outcomes(app, number, code):
    assert code in inquire(signed_in(app), app, number).text


def test_injected_notice_then_acknowledge(app):
    s = signed_in(app)
    requests.post(f"{app}/__control/faults", json={"notice_once": True})
    r = inquire(s, app, "12345")
    assert "SYSTEM NOTICE" in r.text
    r = s.post(f"{app}/notice", data={"f_0012": "12345"})
    assert "MEMBER INQUIRY - DETAIL" in r.text


def test_injected_error_and_expiry(app):
    s = signed_in(app)
    requests.post(f"{app}/__control/faults", json={"error_once": True})
    assert inquire(s, app, "12345").status_code == 500
    requests.post(f"{app}/__control/faults", json={"expire_session": True})
    assert "CL-0401" in inquire(s, app, "12345").text


def test_open_share_validates_then_posts(app):
    s = signed_in(app)
    r = s.post(f"{app}/share/review", data={"m": "12345", "f_0031": "MM", "f_0033": "10"})
    assert "CL-0313" in r.text
    r = s.post(f"{app}/share/review", data={"m": "12345", "f_0031": "SC", "f_0033": "25"})
    assert "SHARE MAINTENANCE - REVIEW" in r.text
    r = s.post(f"{app}/share/post", data={"m": "12345", "f_0031": "SC", "f_0033": "25"})
    assert "OPENED" in r.text


def test_faults_endpoint_disabled_by_default():
    server, url = serve_in_thread(port=0)
    try:
        assert requests.post(f"{url}/__control/faults", json={}).status_code == 404
    finally:
        server.shutdown()
