import { CountUp } from './assets/countUp.min.js'


const div = document.getElementById('why');
const divHeight = div.offsetHeight - window.innerHeight;
const divTop = div.offsetTop;

let activeIndex = 0;
let scrollStopTimeout;

$('#why .reason h2').click(function () {
    const section = $(this).closest('.reason');
    const index = section.index() - 0.8; // prevent rounding errors w/ 1.0
    const sectionTop = divTop + (divHeight * (index / 5));
    $('html, body').animate({
        scrollTop: sectionTop
    }, 500);
})

function scrollHandler() {
    const scrollPosition = window.scrollY;

    if (scrollPosition >= divTop) {
        const relativeScrollPosition = scrollPosition - divTop;
        const percentageScrolled = relativeScrollPosition / divHeight;

        let itemIndex = 0;
        let scrollPercentage = 0;

        if (percentageScrolled > 0.8) {
            itemIndex = 4;
            scrollPercentage = (percentageScrolled - 0.8) / 0.2;
        } else if (percentageScrolled > 0.6) {
            itemIndex = 3;
            scrollPercentage = (percentageScrolled - 0.6) / 0.2;
        } else if (percentageScrolled > 0.4) {
            itemIndex = 2;
            scrollPercentage = (percentageScrolled - 0.4) / 0.2;
        } else if (percentageScrolled > 0.2) {
            itemIndex = 1;
            scrollPercentage = (percentageScrolled - 0.2) / 0.2;
        } else {
            itemIndex = 0;
            scrollPercentage = percentageScrolled / 0.2;
        }

        $('#why .scroll-image').removeClass('active')
        const toShowImage = $('#why .scroll-image').eq(itemIndex)
        toShowImage.addClass('active')

        $('#why .reason').find('.inner').css('width', `${scrollPercentage * 100}%`);
        if (activeIndex != itemIndex) {
            activeIndex = itemIndex;
            $('#why .reason').removeClass('active').find('p').slideUp()
            $('#why .reason').eq(itemIndex).addClass('active').find('p').slideDown();
        }
    }
}

window.addEventListener('scroll', scrollHandler);
scrollHandler()

AOS.init({
    startEvent: 'load',
    once: true
})

document.addEventListener('aos:in:downloads', ({ detail }) => {
    new CountUp('stat-downloads', 300000).start();
});

document.addEventListener('aos:in:citations', ({ detail }) => {
    new CountUp('stat-citations', 30).start();
});

$('#citation').click(function () {
    $(this).toggleClass('open')
    $('#citation-content').toggle()
})

$('#copy').click(function () {
    navigator.clipboard.writeText($('#citation-content pre').text()).then(function () {
        $('#copy').text('copied!')
    }, function (err) {
        alert('An error occurred. You will have to manually copy the text.');
    });
})