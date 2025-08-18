
## "First" Map Posts

<ul>
{% assign firstmap = site.firstmap | sort: 'order' %}
{% for student in firstmap %}
  {% for post in student %}
    <li>
      <a href="{{ post.url }}">{{ student }}</a>
    </li>
  {% endfor %}
{% endfor %}
</ul>
