[[_TOC_]]

# Test

```
npx @modelcontextprotocol/inspector
```

# Inicio

NEO4J_URL=neo4j://neo4j.pm.comsatel.com.pe:7687
NEO4J_USERNAME=neoj4
NEO4J_PASSWORD=welcome1



# Ejemplos

## Episodio: mi familia
Episodio: Mi familia está compuesta por mi esposa Ximena y mi hija Nicole
Fuente: bibliografia personal

# Episodio: arquitectura de microservicios
Episodio: El microservicio se conecta a una base de datos creada especificamente para almacenar la información del dominio implementado por el microservicio.
Source Description (Fuente): metodologia de desarrollo

# Episodio: arquitectura de microservicios
Episodio: Las base de datos donde el microservicio almacena información pueden ser MySQL Server para almacenar datos estructurado o MongoDB para almacenar datos no estructurados (documentos JSON). El microservicio utiliza una base de datos Redis de tipo Key/Value para realizar caching de datos y acelerar el tiempo de respuesta.
Source Description (Fuente): metodologia de desarrollo

# Episodio: arquitectura de microservicios
MySQL Server, Redis y MongoDB son base de datos.
Source Description (Fuente): metodologia de desarrollo

# Episodio: arquitectura de microservicios (frontend)

La arquitectura considera que el frontend web se implementa a través de un portal central y una o varias micro UI desarrolladas con framework Angular.

# Episodio: arquitectura de aplicaciones moviles

{
  "name": "arquitectura de aplicaciones moviles",
  "episode_body": "Las aplicaciones moviles son desarrolladas con framework Flutter para iOS y Android. Se debe utilizar las versiones más usadas de Sistema Operativo al momento de realizarse el desarrollo de la aplicación.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: arquitectura de microservicios (capas)
{
  "name": "arquitectura de microservicio",
  "episode_body": "El microservicio tiene una estructura interna basada en capas (layer). Las capas son: presentation, application, domain, infrastructure. La capa presentation puede acceder a la capa application, la capa application puede acceder a la capa domain e infraestructure",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: arquitectura de microservicios (seguridad)
{
  "name": "arquitectura de microservicio",
  "episode_body": "La capa de presentation expone las funcionalidades a traves de un API Restfull. Los endpoints del API reciben las peticiones y entregan las respuestas a través de documentos JSON. El API es asegurada a través de token JWT basados en OAuth2 y basadas en RBAC. RedHat Keycloak es el producto que ofrece la seguridad de los microservicios.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: arquitectura de microservicios (acceso)
{
  "name": "arquitectura de microservicio",
  "episode_body": "La capa de Presentation puede acceder a la capa de Application y a la capa de infraestructura. La capa de infraestructura puede estar conectada a middleware de mensaje basado en Apache Kafka o RabbitMQ para recibir o enviar notificaciones de eventos.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: arquitectura de microservicios (frameworks)

{
  "name": "arquitectura de microservicio",
  "episode_body": "El framework Spring Boot y Spring Cloud son utilizados en las capas de Presentation, Application, Domain e Infrastructure. El framework Angular es utilizado en los frontends Web.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: arquitectura de microservicios (seguridad)
{
  "name": "arquitectura de microservicio",
  "episode_body": "Se tiene un portafolio de microservicios compuesto por: citas, solicitudes de servicio, vehiculos, tramas altas, contratos, ventas, inventario, unidades, telemetria, servidor de telemetria, notificacion de eventos, gestion documental, comandos.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: arquitectura de microservicios (seguridad)

{
  "name": "arquitectura de microservicio",
  "episode_body": "La capa de Application define la raiz de las transacciones. Las transacciones son Use Cases con limites y reglas bien establecidas. Se aplican patrones (de acuerdo con Domain Driven Design) tales como Service, Factory, Data Transfer Object (DTO), Mapper (transformaciones de datos hacia y desde capa Domain o Infrastructure)",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: arquitectura de microservicios (seguridad)

{
  "name": "arquitectura de microservicio",
  "episode_body": "EL microservicio Comando tiene por proposito permitir gestionar el envío, monitoreo, reenvio y cancelación de comandos enviados a los dispositivos GPS instalados en las unidades.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: Informacion de las unidades

{
  "name": "unidades rastreadas",
  "episode_body": "Las unidades puede ser de tipos diferentes (vehiculo, moto, scooter, persona, producto) que requiera ser rastreada usando dispositivo GPS.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: Información de unidad vehiculoar
{
  "name": "unidades vehiculares",
  "episode_body": "Las unidades vehiculares o simplemente vehiculo tiene como información: motor (obligatorio), chasis (obligatorio), placa (opcional), color, año (de fabricacion), marca, modelo.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: comportamiento de microservicios
{
  "name": "lineamiento sobre microservicios",
  "episode_body": "El microservicio debe responder con Status Code 200 en toda operacion de consulta o actualización exitosa, que sea exitosa, 201 para toda operacion de creacion exitosa. Toda operación que falle por error de datos del consumidor debe responder con Status Code 400. Toda operación que falle por error en el servidor debe responder con Status Code 500. Para todos los casos de falla siempre se debe retornar un codigo (numerico) y mensaje de error personalizado usando jerga propia de negocio, salvo los que se asocien a problemas técnicos.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: información de unidades tipo mascotas

{
  "name": "unidades tipo mascota",
  "episode_body": "Las unidades de tipo mascota tienen como información: nombre, tipo (perro, gato, etc), fecha de nacimiento, raza, talla, peso, color, características distintivas y avatar. Son obligatorios el nombre, tipo y raza.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio: gestion de las unidades

{
  "name": "gestion de las unidades",
  "episode_body": "El area de central, parte de la Gerencia de Operaciones, son los responsables del registro y actualización de las unidades.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

{
  "name": "gestion de las unidades",
  "episode_body": "Las unidades no pueden ser eliminadas sino inactivadas cuando estas ya no cuentan con un contrato activo.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

{
  "name": "gestion de las unidades",
  "episode_body": "Las unidades cuando se registran no tienen asignado ningun contrato ni tampoco instalado ningun equipo (GPS, camara dashcam) o accesorio (boton de panico, relay). Las unidades tienen un propietario.",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}

# Episodio:

{
  "name": "gestion de las unidades",
  "episode_body": "Las unidades al ser registradas no tendrán un dispositivo equipos (GPS,dashcam) o accesorios instalados. El registro de dispositivo y accesorios instalados en la unidad se registran durante la aprobación de las cita de instación o cita de mantenimiento. En las cita de retiro se retiran de la unidad equipos o los accesorios según la solicitud de servicio (pueden ser de tipo instalación, mantenimiento o retiro)",
  "group_id": "arquitectura",
  "source_description": "metodologia"
}